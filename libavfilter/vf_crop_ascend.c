/*
 * Ascend NPU (CANN VPC) hardware cropping filter.
 *
 * Copyright (c) 2026 green-hand-shandong
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>

#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_ascend.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "scale_eval.h"
#include "libavutil/eval.h"
#include "video.h"

typedef struct CropAscendContext {
    const AVClass *class;

    int x, y;            /* crop origin */
    char *w_expr, *h_expr, *x_expr, *y_expr;
    int out_w, out_h;

    AVBufferRef *frames_ref;

    aclrtContext context;
    hi_vpc_chn vpc_chn;
    int vpc_created;
    int dev_id;
} CropAscendContext;

static int crop_ascend_query_formats(const AVFilterContext *avctx,
                                     AVFilterFormatsConfig **cfg_in,
                                     AVFilterFormatsConfig **cfg_out)
{
    return ff_set_common_formats2(avctx, cfg_in, cfg_out,
                                  ff_make_formats_list_singleton(AV_PIX_FMT_ASCEND));
}

static void crop_ascend_uninit(AVFilterContext *avctx)
{
    CropAscendContext *ctx = avctx->priv;

    if (ctx->vpc_created) {
        hi_mpi_vpc_destroy_chn(ctx->vpc_chn);
        ctx->vpc_created = 0;
    }
    if (ctx->context) {
        aclrtDestroyContext(ctx->context);
        ctx->context = NULL;
    }
    av_buffer_unref(&ctx->frames_ref);
}

/* Shared by both config_output paths: bind device, init VPC channel. */
static int crop_bind_device(AVFilterContext *avctx, AVBufferRef *device_ref)
{
    CropAscendContext *ctx = avctx->priv;
    AVAscendDeviceContext *dev =
        (AVAscendDeviceContext *)((AVHWDeviceContext *)device_ref->data)->hwctx;
    hi_vpc_chn_attr attr = { 0 };
    int ret;

    ctx->dev_id = dev->device_id;
    ret = aclrtSetDevice(dev->device_id);
    if (ret == 0)
        ret = aclrtCreateContext(&ctx->context, dev->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "crop_ascend: context setup failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }
    aclrtSetCurrentContext(ctx->context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "crop_ascend: sys_init failed: 0x%x\n", ret);
        return AVERROR_EXTERNAL;
    }

    ret = hi_mpi_vpc_sys_create_chn(&ctx->vpc_chn, &attr);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "crop_ascend: vpc create_chn failed: 0x%x\n", ret);
        return AVERROR_EXTERNAL;
    }
    ctx->vpc_created = 1;
    return 0;
}

static int crop_ascend_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    CropAscendContext *ctx = avctx->priv;
    AVFilterLink *inlink = avctx->inputs[0];
    FilterLink *in_fl = ff_filter_link(inlink);
    FilterLink *out_fl = ff_filter_link(outlink);
    AVHWFramesContext *in_frames;
    AVHWFramesContext *fctx;
    int ret;

    if (!in_fl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "crop_ascend: input frames have no hw context\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)in_fl->hw_frames_ctx->data;

    /* w/h via the scale evaluator (supports in_w/out_w etc); x/y via
     * plain expression eval against in/out sizes. */
    ret = ff_scale_eval_dimensions(ctx, ctx->w_expr, ctx->h_expr,
                                   inlink, outlink, &ctx->out_w, &ctx->out_h);
    if (ret < 0)
        return ret;
    {
        static const char *const var_names[] = { "in_w", "in_h", "out_w",
                                                 "out_h", NULL };
        double var_values[] = { inlink->w, inlink->h, ctx->out_w, ctx->out_h };
        AVExpr *e = NULL;
        double res;

        ret = av_expr_parse(&e, ctx->x_expr, var_names, NULL, NULL,
                            NULL, NULL, 0, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "crop_ascend: bad x expr '%s'\n", ctx->x_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->x = (int)res;

        ret = av_expr_parse(&e, ctx->y_expr, var_names, NULL, NULL,
                            NULL, NULL, 0, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "crop_ascend: bad y expr '%s'\n", ctx->y_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->y = (int)res;
    }

    if (ctx->x < 0 || ctx->y < 0 ||
        ctx->x + ctx->out_w > inlink->w || ctx->y + ctx->out_h > inlink->h) {
        av_log(avctx, AV_LOG_ERROR,
               "crop_ascend: region %dx%d@(%d,%d) outside %dx%d input\n",
               ctx->out_w, ctx->out_h, ctx->x, ctx->y, inlink->w, inlink->h);
        return AVERROR(EINVAL);
    }

    outlink->w = ctx->out_w;
    outlink->h = ctx->out_h;

    if (!ctx->frames_ref) {
        ctx->frames_ref = av_hwframe_ctx_alloc(in_frames->device_ref);
        if (!ctx->frames_ref)
            return AVERROR(ENOMEM);
        fctx = (AVHWFramesContext *)ctx->frames_ref->data;
        fctx->format    = AV_PIX_FMT_ASCEND;
        fctx->sw_format = in_frames->sw_format;
        fctx->width     = FFALIGN(ctx->out_w, 16);
        fctx->height    = FFALIGN(ctx->out_h, 2);
        fctx->initial_pool_size = 8;
        ret = av_hwframe_ctx_init(ctx->frames_ref);
        if (ret < 0) {
            av_buffer_unref(&ctx->frames_ref);
            return ret;
        }
    }
    out_fl->hw_frames_ctx = av_buffer_ref(ctx->frames_ref);
    if (!out_fl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    ret = crop_bind_device(avctx, in_frames->device_ref);
    if (ret < 0)
        return ret;

    /* 310P VPC crop region: offsets and size must be even for NV12. */
    if ((ctx->x & 1) || (ctx->y & 1) || (ctx->out_w & 1) || (ctx->out_h & 1)) {
        av_log(avctx, AV_LOG_ERROR,
               "crop_ascend: region must be even-aligned (got %dx%d@%d,%d)\n",
               ctx->out_w, ctx->out_h, ctx->x, ctx->y);
        return AVERROR(EINVAL);
    }
    return 0;
}

static int crop_ascend_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *avctx = inlink->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    CropAscendContext *ctx = avctx->priv;
    AVFrame *out = NULL;
    hi_vpc_pic_info src = { 0 }, dst = { 0 };
    hi_vpc_crop_region_info crop_info = { 0 };
    hi_u32 task_id = 0;
    int ret;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;

    src.picture_address       = in->data[0];
    src.picture_width         = in->width;
    src.picture_height        = in->height;
    src.picture_width_stride  = in->linesize[0] > 0 ? in->linesize[0] :
                                FFALIGN(in->width, 16);
    src.picture_height_stride = FFALIGN(in->height, 2);
    src.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    src.picture_buffer_size   = (size_t)src.picture_width_stride *
                                src.picture_height_stride * 3 / 2;

    dst.picture_address       = out->data[0];
    dst.picture_width         = out->width;
    dst.picture_height        = out->height;
    dst.picture_width_stride  = out->linesize[0] > 0 ? out->linesize[0] :
                                FFALIGN(out->width, 16);
    dst.picture_height_stride = FFALIGN(out->height, 2);
    dst.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    dst.picture_buffer_size   = (size_t)dst.picture_width_stride *
                                dst.picture_height_stride * 3 / 2;

    crop_info.crop_region.top_offset  = ctx->y;
    crop_info.crop_region.left_offset = ctx->x;
    crop_info.crop_region.crop_width  = ctx->out_w;
    crop_info.crop_region.crop_height = ctx->out_h;
    crop_info.dest_pic_info = dst;

    aclrtSetCurrentContext(ctx->context);
    ret = hi_mpi_vpc_crop(ctx->vpc_chn, &src, &crop_info, 1, &task_id, 0);
    if (ret == 0)
        ret = hi_mpi_vpc_get_process_result(ctx->vpc_chn, task_id, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "crop_ascend: vpc_crop failed: 0x%x\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(CropAscendContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption crop_ascend_options[] = {
    { "w", "crop width expression", OFFSET(w_expr),
      AV_OPT_TYPE_STRING, { .str = "in_w/2" }, 0, 0, FLAGS },
    { "h", "crop height expression", OFFSET(h_expr),
      AV_OPT_TYPE_STRING, { .str = "in_h/2" }, 0, 0, FLAGS },
    { "x", "crop x offset expression", OFFSET(x_expr),
      AV_OPT_TYPE_STRING, { .str = "(in_w-out_w)/2" }, 0, 0, FLAGS },
    { "y", "crop y offset expression", OFFSET(y_expr),
      AV_OPT_TYPE_STRING, { .str = "(in_h-out_h)/2" }, 0, 0, FLAGS },
    { "device_id", "Ascend NPU device id", OFFSET(dev_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(crop_ascend);

static const AVFilterPad crop_ascend_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &crop_ascend_filter_frame,
    },
};

static const AVFilterPad crop_ascend_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &crop_ascend_config_output,
    },
};

const FFFilter ff_vf_crop_ascend = {
    .p.name        = "crop_ascend",
    .p.description = NULL_IF_CONFIG_SMALL("Crop video on Ascend NPU VPC."),
    .p.priv_class  = &crop_ascend_class,
    .p.flags       = AVFILTER_FLAG_HWDEVICE,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .priv_size     = sizeof(CropAscendContext),
    .uninit        = &crop_ascend_uninit,
    FILTER_INPUTS(crop_ascend_inputs),
    FILTER_OUTPUTS(crop_ascend_outputs),
    FILTER_QUERY_FUNC2(crop_ascend_query_formats),
};
