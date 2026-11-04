/*
 * Ascend NPU (CANN VPC) hardware scaling filter.
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
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "scale_eval.h"
#include "video.h"

typedef struct ScaleAscendContext {
    const AVClass *class;

    int device_id;
    int interpolation;   /* 0: bilinear, 1: nearest */
    char *w_expr;
    char *h_expr;
    char *size_str;

    int out_w, out_h;
    int passthrough;

    AVBufferRef *frames_ref;

    aclrtContext context;
    int dev_id;
    aclrtContext dev_context;
    hi_vpc_chn vpc_chn;
    int vpc_created;
} ScaleAscendContext;

static int ascend_scale_query_formats(const AVFilterContext *avctx,
                                      AVFilterFormatsConfig **cfg_in,
                                      AVFilterFormatsConfig **cfg_out)
{
    return ff_set_common_formats2(avctx, cfg_in, cfg_out,
                                  ff_make_formats_list_singleton(AV_PIX_FMT_ASCEND));
}

static int ascend_scale_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    ScaleAscendContext *ctx = avctx->priv;
    AVFilterLink *inlink = avctx->inputs[0];
    FilterLink *in_fl = ff_filter_link(inlink);
    FilterLink *out_fl = ff_filter_link(outlink);
    AVHWFramesContext *in_frames;
    AVHWFramesContext *fctx;
    AVAscendDeviceContext *dev;
    int ret;

    if (!in_fl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "scale_ascend: input frames have no hw context\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)in_fl->hw_frames_ctx->data;
    if (in_frames->format != AV_PIX_FMT_ASCEND) {
        av_log(avctx, AV_LOG_ERROR, "scale_ascend: unsupported input %s\n",
               av_get_pix_fmt_name(in_frames->format));
        return AVERROR(EINVAL);
    }

    ret = ff_scale_eval_dimensions(ctx, ctx->w_expr, ctx->h_expr,
                                   inlink, outlink,
                                   &ctx->out_w, &ctx->out_h);
    if (ret < 0)
        return ret;
    ff_scale_adjust_dimensions(inlink, &ctx->out_w, &ctx->out_h,
                               0, 2, 2);
    /* Even at same size the frame must go through VPC: passing frames of
     * a foreign pool straight through mixes buffer lifetimes across the
     * decoder/encoder pools at stream teardown. */
    ctx->passthrough = 0;

    outlink->w = ctx->out_w;
    outlink->h = ctx->out_h;
    outlink->format = AV_PIX_FMT_ASCEND;
    ff_filter_link(outlink)->frame_rate = ff_filter_link(inlink)->frame_rate;

    av_buffer_unref(&ctx->frames_ref);
    ctx->frames_ref = av_hwframe_ctx_alloc(in_frames->device_ref);
    if (!ctx->frames_ref)
        return AVERROR(ENOMEM);
    fctx = (AVHWFramesContext *)ctx->frames_ref->data;
    fctx->format    = AV_PIX_FMT_ASCEND;
    fctx->sw_format = in_frames->sw_format;
    fctx->width     = ctx->out_w;
    fctx->height    = ctx->out_h;
    fctx->initial_pool_size = 16;
    ret = av_hwframe_ctx_init(ctx->frames_ref);
    if (ret < 0) {
        av_buffer_unref(&ctx->frames_ref);
        return ret;
    }
    out_fl->hw_frames_ctx = av_buffer_ref(ctx->frames_ref);
    if (!out_fl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    /* VPC channel on the input device: reuse the device's own context
     * (the hwdevice already initialized ACL + DVPP); creating a separate
     * context here would leave the VPC channel invisible on the device
     * context the frames were allocated in. */
    dev = (AVAscendDeviceContext *)((AVHWDeviceContext *)in_frames->device_ref->data)->hwctx;
    ctx->context = NULL; /* use the device default via SetDevice */

    ret = aclrtSetDevice(dev->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "scale_ascend: aclrtSetDevice failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    {
        hi_vpc_chn_attr attr = { 0 };

        /* attr is the task queue depth; size it generously since frames
         * arrive as fast as the decoder can produce them. */
        attr.attr       = 32;
        attr.pic_width  = FFALIGN(inlink->w, 16);
        attr.pic_height = FFALIGN(inlink->h, 16);
        ret = hi_mpi_vpc_sys_create_chn(&ctx->vpc_chn, &attr);
        if (ret != 0) {
            av_log(avctx, AV_LOG_ERROR, "scale_ascend: vpc create_chn failed: 0x%x\n", ret);
            return AVERROR_EXTERNAL;
        }
        ctx->vpc_created = 1;
    }

    return 0;
}

static int ascend_scale_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *avctx = inlink->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    ScaleAscendContext *ctx = avctx->priv;
    AVFrame *out = NULL;
    hi_vpc_pic_info src = { 0 }, dst = { 0 };
    hi_u32 task_id = 0;
    int ret;

    if (ctx->passthrough)
        return ff_filter_frame(outlink, in);

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

    /* filter threads are foreign: bind the device context every call. */
    aclrtSetCurrentContext(ctx->dev_context);
    ret = hi_mpi_vpc_resize(ctx->vpc_chn, &src, &dst, 0.0, 0.0,
                            ctx->interpolation, &task_id, 0);
    if (ret == 0)
        ret = hi_mpi_vpc_get_process_result(ctx->vpc_chn, task_id, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "scale_ascend: vpc_resize failed: 0x%x\n", ret);
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

static av_cold int ascend_scale_uninit(AVFilterContext *avctx)
{
    ScaleAscendContext *ctx = avctx->priv;

    if (ctx->vpc_created) {
        hi_mpi_vpc_destroy_chn(ctx->vpc_chn);
        ctx->vpc_created = 0;
    }
    av_buffer_unref(&ctx->frames_ref);
    return 0;
}

#define OFFSET(x) offsetof(ScaleAscendContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption scale_ascend_options[] = {
    { "w", "output width expression (-1 to keep)", OFFSET(w_expr),
      AV_OPT_TYPE_STRING, { .str = "-1" }, 0, 0, FLAGS },
    { "h", "output height expression (-1 to keep)", OFFSET(h_expr),
      AV_OPT_TYPE_STRING, { .str = "-1" }, 0, 0, FLAGS },
    { "s", "output size WxH", OFFSET(size_str),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "interp", "interpolation method", OFFSET(interpolation),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1, FLAGS, .unit = "interp" },
    { "bilinear", "bilinear",  0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, .unit = "interp" },
    { "nearest",  "nearest",   0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, .unit = "interp" },
    { "device_id", "Ascend NPU device id", OFFSET(device_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(scale_ascend);

static const AVFilterPad ascend_scale_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &ascend_scale_filter_frame,
    },
};

static const AVFilterPad ascend_scale_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ascend_scale_config_output,
    },
};

const FFFilter ff_vf_scale_ascend = {
    .p.name        = "scale_ascend",
    .p.description = NULL_IF_CONFIG_SMALL("Scale video on Ascend NPU VPC."),
    .p.priv_class  = &scale_ascend_class,
    .p.flags       = AVFILTER_FLAG_HWDEVICE,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .priv_size     = sizeof(ScaleAscendContext),
    .uninit        = &ascend_scale_uninit,
    FILTER_INPUTS(ascend_scale_inputs),
    FILTER_OUTPUTS(ascend_scale_outputs),
    FILTER_QUERY_FUNC2(ascend_scale_query_formats),
};
