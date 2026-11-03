/*
 * Ascend NPU (CANN VPC) hardware padding (letterbox) filter.
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

typedef struct PadAscendContext {
    const AVClass *class;

    char *w_expr, *h_expr, *x_expr, *y_expr;
    char *color_str;
    int out_w, out_h, pad_x, pad_y;
    int color_y, color_u, color_v;

    AVBufferRef *frames_ref;

    aclrtContext context;
    hi_vpc_chn vpc_chn;
    int vpc_created;
    int dev_id;
} PadAscendContext;

static int pad_ascend_query_formats(const AVFilterContext *avctx,
                                    AVFilterFormatsConfig **cfg_in,
                                    AVFilterFormatsConfig **cfg_out)
{
    return ff_set_common_formats2(avctx, cfg_in, cfg_out,
                                  ff_make_formats_list_singleton(AV_PIX_FMT_ASCEND));
}

static av_cold void pad_ascend_uninit(AVFilterContext *avctx)
{
    PadAscendContext *ctx = avctx->priv;

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

/* color option: 0xYYYYUUUVV (like vf_pad) or named; we accept the hex form. */
static int pad_parse_color(const char *arg, int *y, int *u, int *v)
{
    unsigned long v32;

    if (!arg || !arg[0])
        return AVERROR(EINVAL);
    v32 = strtoul(arg, NULL, 0);
    if (v32 <= 0xFF) {
        *y = (int)v32; *u = 128; *v = 128;      /* gray level */
    } else if (v32 <= 0xFFFF) {
        *y = (int)(v32 >> 8) & 0xFF;
        *u = (int)v32 & 0xFF; *v = *u;
    } else {
        *y = (int)(v32 >> 16) & 0xFF;
        *u = (int)(v32 >> 8) & 0xFF;
        *v = (int)v32 & 0xFF;
    }
    return 0;
}

static int pad_ascend_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    PadAscendContext *ctx = avctx->priv;
    AVFilterLink *inlink = avctx->inputs[0];
    FilterLink *in_fl = ff_filter_link(inlink);
    FilterLink *out_fl = ff_filter_link(outlink);
    AVHWFramesContext *in_frames;
    AVHWFramesContext *fctx;
    int ret;

    if (!in_fl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "pad_ascend: input frames have no hw context\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)in_fl->hw_frames_ctx->data;

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
            av_log(avctx, AV_LOG_ERROR, "pad_ascend: bad x expr '%s'\n", ctx->x_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->pad_x = (int)res;

        ret = av_expr_parse(&e, ctx->y_expr, var_names, NULL, NULL,
                            NULL, NULL, 0, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "pad_ascend: bad y expr '%s'\n", ctx->y_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->pad_y = (int)res;
    }

    if (ctx->pad_x < 0 || ctx->pad_y < 0 ||
        ctx->pad_x + inlink->w > ctx->out_w ||
        ctx->pad_y + inlink->h > ctx->out_h) {
        av_log(avctx, AV_LOG_ERROR,
               "pad_ascend: content %dx%d@(%d,%d) does not fit %dx%d\n",
               inlink->w, inlink->h, ctx->pad_x, ctx->pad_y,
               ctx->out_w, ctx->out_h);
        return AVERROR(EINVAL);
    }
    if ((ctx->out_w & 1) || (ctx->out_h & 1) || (ctx->pad_x & 1) ||
        (ctx->pad_y & 1)) {
        av_log(avctx, AV_LOG_ERROR, "pad_ascend: sizes/offsets must be even\n");
        return AVERROR(EINVAL);
    }

    if (pad_parse_color(ctx->color_str, &ctx->color_y, &ctx->color_u,
                        &ctx->color_v) < 0) {
        /* default black in limited-range YUV */
        ctx->color_y = 16; ctx->color_u = 128; ctx->color_v = 128;
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

    {
        AVAscendDeviceContext *dev =
            (AVAscendDeviceContext *)((AVHWDeviceContext *)in_frames->device_ref->data)->hwctx;
        hi_vpc_chn_attr attr = { 0 };
        int arc;

        ctx->dev_id = dev->device_id;
        arc = aclrtSetDevice(dev->device_id);
        if (arc == 0)
            arc = aclrtCreateContext(&ctx->context, dev->device_id);
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "pad_ascend: context setup failed: %d\n", arc);
            return AVERROR_EXTERNAL;
        }
        aclrtSetCurrentContext(ctx->context);

        arc = hi_mpi_sys_init();
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "pad_ascend: sys_init failed: 0x%x\n", arc);
            return AVERROR_EXTERNAL;
        }

        arc = hi_mpi_vpc_sys_create_chn(&ctx->vpc_chn, &attr);
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "pad_ascend: vpc create_chn failed: 0x%x\n", arc);
            return AVERROR_EXTERNAL;
        }
        ctx->vpc_created = 1;
    }
    return 0;
}

static int pad_ascend_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *avctx = inlink->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    PadAscendContext *ctx = avctx->priv;
    AVFrame *out = NULL;
    hi_vpc_pic_info src = { 0 }, dst = { 0 };
    hi_vpc_make_border_info bi = { 0 };
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

    bi.top         = ctx->pad_y;
    bi.bottom      = ctx->out_h - ctx->pad_y - in->height;
    bi.left        = ctx->pad_x;
    bi.right       = ctx->out_w - ctx->pad_x - in->width;
    bi.border_type = HI_BORDER_CONSTANT;
    /* NV12 scalar: val[0]=Y, val[1]=U, val[2]=V (doubles) */
    bi.scalar_value.val[0] = ctx->color_y;
    bi.scalar_value.val[1] = ctx->color_u;
    bi.scalar_value.val[2] = ctx->color_v;
    bi.scalar_value.val[3] = 0;

    aclrtSetCurrentContext(ctx->context);
    ret = hi_mpi_vpc_copy_make_border(ctx->vpc_chn, &src, &dst, bi, &task_id, 0);
    if (ret == 0)
        ret = hi_mpi_vpc_get_process_result(ctx->vpc_chn, task_id, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "pad_ascend: copy_make_border failed: 0x%x\n", ret);
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

#define OFFSET(x) offsetof(PadAscendContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption pad_ascend_options[] = {
    { "w", "output width expression", OFFSET(w_expr),
      AV_OPT_TYPE_STRING, { .str = "in_w" }, 0, 0, FLAGS },
    { "h", "output height expression", OFFSET(h_expr),
      AV_OPT_TYPE_STRING, { .str = "in_h" }, 0, 0, FLAGS },
    { "x", "content x offset expression", OFFSET(x_expr),
      AV_OPT_TYPE_STRING, { .str = "(out_w-in_w)/2" }, 0, 0, FLAGS },
    { "y", "content y offset expression", OFFSET(y_expr),
      AV_OPT_TYPE_STRING, { .str = "(out_h-in_h)/2" }, 0, 0, FLAGS },
    { "color", "border color (0xYYYYUUUVV or gray level)", OFFSET(color_str),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "device_id", "Ascend NPU device id", OFFSET(dev_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(pad_ascend);

static const AVFilterPad pad_ascend_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &pad_ascend_filter_frame,
    },
};

static const AVFilterPad pad_ascend_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &pad_ascend_config_output,
    },
};

const FFFilter ff_vf_pad_ascend = {
    .p.name        = "pad_ascend",
    .p.description = NULL_IF_CONFIG_SMALL("Pad video on Ascend NPU VPC (letterbox)."),
    .p.priv_class  = &pad_ascend_class,
    .p.flags       = AVFILTER_FLAG_HWDEVICE,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .priv_size     = sizeof(PadAscendContext),
    .uninit        = &pad_ascend_uninit,
    FILTER_INPUTS(pad_ascend_inputs),
    FILTER_OUTPUTS(pad_ascend_outputs),
    FILTER_QUERY_FUNC2(pad_ascend_query_formats),
};
