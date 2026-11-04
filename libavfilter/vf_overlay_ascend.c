/*
 * Ascend NPU (CANN VPC) hardware overlay filter.
 *
 * Overlays the second (overlay) input onto the first (main) input using
 * VPC crop_resize_paste on the NPU. Dual-input synchronization uses the
 * standard framesync dualinput mode (same as overlay_cuda).
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

#include "libavutil/eval.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_ascend.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"
#include "video.h"

typedef struct OverlayAscendContext {
    const AVClass *class;

    char *x_expr, *y_expr;
    int x, y;

    AVBufferRef *frames_ref;

    aclrtContext context;
    hi_vpc_chn vpc_chn;
    int vpc_created;
    int dev_id;
    aclrtContext dev_context;

    FFFrameSync fs;
} OverlayAscendContext;

static int overlay_ascend_filter_frame(FFFrameSync *fs);

static av_cold int overlay_ascend_init(AVFilterContext *avctx)
{
    OverlayAscendContext *ctx = avctx->priv;

    /* on_event must be registered before the first framesync activate;
     * doing it in init() keeps it set across graph reinits. */
    ctx->fs.on_event = &overlay_ascend_filter_frame;
    return 0;
}

static int overlay_ascend_query_formats(const AVFilterContext *avctx,
                                        AVFilterFormatsConfig **cfg_in,
                                        AVFilterFormatsConfig **cfg_out)
{
    return ff_set_common_formats2(avctx, cfg_in, cfg_out,
                                  ff_make_formats_list_singleton(AV_PIX_FMT_ASCEND));
}

static av_cold void overlay_ascend_uninit(AVFilterContext *avctx)
{
    OverlayAscendContext *ctx = avctx->priv;

    ff_framesync_uninit(&ctx->fs);
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

/* Compose: copy main onto dst (D2D), then VPC-paste the overlay at (x,y). */
static int overlay_ascend_blend(AVFilterContext *avctx, OverlayAscendContext *ctx,
                                aclrtContext dev_context,
                                AVFrame *main_f, AVFrame *ovl_f, AVFrame *out)
{
    hi_vpc_pic_info src = { 0 }, dst = { 0 };
    hi_vpc_crop_resize_paste_region paste = { 0 };
    hi_u32 task_id = 0;
    int ret;

    /* Device context was created on the config thread; the activate
     * thread may run later (or after a graph re-init). Always re-bind
     * by device id and keep a fresh context handle. */
    aclrtSetDevice(ctx->dev_id);
    aclrtSetCurrentContext(dev_context);
    av_log(avctx, AV_LOG_WARNING, "overlay_ascend: blend main=%p ovl=%p out=%p\n",
           main_f->data[0], ovl_f->data[0], out->data[0]);

    dst.picture_address       = out->data[0];
    dst.picture_width         = out->width;
    dst.picture_height        = out->height;
    dst.picture_width_stride  = out->linesize[0] > 0 ? out->linesize[0] :
                                FFALIGN(out->width, 16);
    dst.picture_height_stride = FFALIGN(out->height, 2);
    dst.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    dst.picture_buffer_size   = (size_t)dst.picture_width_stride *
                                dst.picture_height_stride * 3 / 2;

    /* Step 1: VPC-copy main 1:1 into out (all inside VPC, no D2D memcpy
     * which would need a different thread-context arrangement). */
    {
        hi_vpc_pic_info msrc = { 0 };
        hi_vpc_crop_region_info cp = { 0 };

        msrc.picture_address       = main_f->data[0];
        msrc.picture_width         = main_f->width;
        msrc.picture_height        = main_f->height;
        msrc.picture_width_stride  = main_f->linesize[0] > 0 ? main_f->linesize[0] :
                                     FFALIGN(main_f->width, 16);
        msrc.picture_height_stride = FFALIGN(main_f->height, 2);
        msrc.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        msrc.picture_buffer_size   = (size_t)msrc.picture_width_stride *
                                     msrc.picture_height_stride * 3 / 2;

        cp.crop_region.top_offset  = 0;
        cp.crop_region.left_offset = 0;
        cp.crop_region.crop_width  = main_f->width;
        cp.crop_region.crop_height = main_f->height;
        cp.dest_pic_info           = dst;

        ret = hi_mpi_vpc_crop(ctx->vpc_chn, &msrc, &cp, 1, &task_id, 0);
        if (ret == 0)
            ret = hi_mpi_vpc_get_process_result(ctx->vpc_chn, task_id, 1000);
        if (ret != 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: main vpc copy failed: 0x%x\n", ret);
            return AVERROR_EXTERNAL;
        }
    }

    /* Step 2: paste the overlay at (x,y). */
    src.picture_address       = ovl_f->data[0];
    src.picture_width         = ovl_f->width;
    src.picture_height        = ovl_f->height;
    src.picture_width_stride  = ovl_f->linesize[0] > 0 ? ovl_f->linesize[0] :
                                FFALIGN(ovl_f->width, 16);
    src.picture_height_stride = FFALIGN(ovl_f->height, 2);
    src.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    src.picture_buffer_size   = (size_t)src.picture_width_stride *
                                src.picture_height_stride * 3 / 2;


    paste.crop_region.top_offset  = 0;
    paste.crop_region.left_offset = 0;
    paste.crop_region.crop_width  = ovl_f->width;
    paste.crop_region.crop_height = ovl_f->height;
    paste.dest_pic_info           = dst;
    paste.dest_top_offset         = ctx->y;
    paste.dest_left_offset        = ctx->x;
    paste.resize_info.resize_width  = ovl_f->width;
    paste.resize_info.resize_height = ovl_f->height;

    ret = hi_mpi_vpc_crop_resize_paste(ctx->vpc_chn, &src, &paste, 1,
                                       &task_id, 0);
    if (ret == 0)
        ret = hi_mpi_vpc_get_process_result(ctx->vpc_chn, task_id, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "overlay_ascend: paste failed: 0x%x\n", ret);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

static int overlay_ascend_filter_frame(FFFrameSync *fs)
{
    AVFilterContext *avctx = fs->parent;
    AVFilterContext *avctx2 = fs->parent;
    OverlayAscendContext *ctx = avctx2->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFrame *main_f = NULL, *ovl_f = NULL, *out = NULL;
    int ret;

    ret = ff_framesync_dualinput_get(fs, &main_f, &ovl_f);
    if (ret < 0)
        return ret;
    if (!main_f)
        return AVERROR_BUG;
    av_log(fs->parent, AV_LOG_WARNING, "overlay_ascend: dualinput got main=%p ovl=%p\n",
           main_f->data[0], ovl_f ? ovl_f->data[0] : NULL);
    if (!ovl_f)
        return ff_filter_frame(outlink, main_f);

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&main_f);
        av_frame_free(&ovl_f);
        return AVERROR(ENOMEM);
    }
    ret = av_frame_copy_props(out, main_f);
    if (ret < 0)
        goto fail;
    ret = overlay_ascend_blend(avctx, ctx, ctx->dev_context, main_f, ovl_f, out);
    if (ret < 0)
        goto fail;

    av_frame_free(&main_f);
    av_frame_free(&ovl_f);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&main_f);
    av_frame_free(&ovl_f);
    av_frame_free(&out);
    return ret;
}

static int overlay_ascend_activate(AVFilterContext *avctx)
{
    OverlayAscendContext *ctx = avctx->priv;
    int r;

    r = ff_framesync_activate(&ctx->fs);
    return r;
}

static int overlay_ascend_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    OverlayAscendContext *ctx = avctx->priv;
    AVFilterLink *mainlink = avctx->inputs[0];
    FilterLink *main_fl = ff_filter_link(mainlink);
    FilterLink *out_fl = ff_filter_link(outlink);
    AVHWFramesContext *in_frames;
    AVHWFramesContext *fctx;
    AVFilterLink *ovlink = avctx->inputs[1];
    int ret;

    av_log(avctx, AV_LOG_WARNING, "overlay_ascend: config_output enter\n");

    if (!main_fl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "overlay_ascend: main input has no hw context\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)main_fl->hw_frames_ctx->data;

        outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->format = AV_PIX_FMT_ASCEND;
        outlink->time_base = mainlink->time_base;
    ff_filter_link(outlink)->frame_rate = main_fl->frame_rate;

    /* evaluate position */
    {
        static const char *const var_names[] = { "main_w", "main_h",
                                                 "overlay_w", "overlay_h", NULL };
        double var_values[] = { mainlink->w, mainlink->h,
                                ovlink ? ovlink->w : 0, ovlink ? ovlink->h : 0 };
        AVExpr *e = NULL;
        double res;

        ret = av_expr_parse(&e, ctx->x_expr, var_names, NULL, NULL,
                            NULL, NULL, 0, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: bad x expr '%s'\n", ctx->x_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->x = (int)res;

        ret = av_expr_parse(&e, ctx->y_expr, var_names, NULL, NULL,
                            NULL, NULL, 0, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: bad y expr '%s'\n", ctx->y_expr);
            return ret;
        }
        res = av_expr_eval(e, var_values, NULL);
        av_expr_free(e);
        ctx->y = (int)res;
    }

    if (ctx->x < 0 || ctx->y < 0 ||
        ctx->x + (ovlink ? ovlink->w : 0) > outlink->w ||
        ctx->y + (ovlink ? ovlink->h : 0) > outlink->h) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_ascend: overlay %dx%d@(%d,%d) does not fit %dx%d\n",
               ovlink ? ovlink->w : 0, ovlink ? ovlink->h : 0, ctx->x, ctx->y,
               outlink->w, outlink->h);
        return AVERROR(EINVAL);
    }

    av_log(avctx, AV_LOG_WARNING, "overlay_ascend: geom x=%d y=%d ok\n", ctx->x, ctx->y);
        if (!ctx->frames_ref) {
        ctx->frames_ref = av_hwframe_ctx_alloc(in_frames->device_ref);
        if (!ctx->frames_ref)
            return AVERROR(ENOMEM);
        fctx = (AVHWFramesContext *)ctx->frames_ref->data;
        fctx->format    = AV_PIX_FMT_ASCEND;
        fctx->sw_format = in_frames->sw_format;
        fctx->width     = FFALIGN(outlink->w, 16);
        fctx->height    = FFALIGN(outlink->h, 2);
        fctx->initial_pool_size = 4;
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
        ctx->dev_context = dev->context;
        arc = aclrtSetDevice(dev->device_id);
        if (arc == 0)
            arc = aclrtCreateContext(&ctx->context, dev->device_id);
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: context setup failed: %d\n", arc);
            return AVERROR_EXTERNAL;
        }
        aclrtSetCurrentContext(ctx->context);

        arc = hi_mpi_sys_init();
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: sys_init failed: 0x%x\n", arc);
            return AVERROR_EXTERNAL;
        }

        arc = hi_mpi_vpc_sys_create_chn(&ctx->vpc_chn, &attr);
        if (arc != 0) {
            av_log(avctx, AV_LOG_ERROR, "overlay_ascend: vpc create_chn failed: 0x%x\n", arc);
            return AVERROR_EXTERNAL;
        }
        ctx->vpc_created = 1;
    }

    av_log(avctx, AV_LOG_WARNING, "overlay_ascend: vpc chn created\n");

    /* framesync dualinput: both inputs hardware frames */
        ret = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (ret < 0)
        return ret;
    ctx->fs.on_event = &overlay_ascend_filter_frame;
    ret = ff_framesync_configure(&ctx->fs);
    return ret;
}

#define OFFSET(x) offsetof(OverlayAscendContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption overlay_ascend_options[] = {
    { "x", "overlay x expression", OFFSET(x_expr),
      AV_OPT_TYPE_STRING, { .str = "0" }, 0, 0, FLAGS },
    { "y", "overlay y expression", OFFSET(y_expr),
      AV_OPT_TYPE_STRING, { .str = "0" }, 0, 0, FLAGS },
    { "device_id", "Ascend NPU device id", OFFSET(dev_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(overlay_ascend);

static int overlay_ascend_config_main(AVFilterLink *inlink)
{
    av_log(inlink->dst, AV_LOG_WARNING, "overlay_ascend: config main %dx%d\n",
           inlink->w, inlink->h);
    return 0;
}

static int overlay_ascend_config_overlay(AVFilterLink *inlink)
{
    av_log(inlink->dst, AV_LOG_WARNING, "overlay_ascend: config overlay %dx%d\n",
           inlink->w, inlink->h);
    return 0;
}

static const AVFilterPad overlay_ascend_inputs[] = {
    {
        .name          = "main",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &overlay_ascend_config_main,
    },
    {
        .name          = "overlay",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &overlay_ascend_config_overlay,
    },
};

static const AVFilterPad overlay_ascend_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &overlay_ascend_config_output,
    },
};

const FFFilter ff_vf_overlay_ascend = {
    .p.name        = "overlay_ascend",
    .p.description = NULL_IF_CONFIG_SMALL("Overlay a video on top of another on Ascend NPU VPC."),
    .p.priv_class  = &overlay_ascend_class,
    .p.flags       = AVFILTER_FLAG_HWDEVICE,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .priv_size     = sizeof(OverlayAscendContext),
    .init          = &overlay_ascend_init,
    .uninit        = &overlay_ascend_uninit,
    .activate      = &overlay_ascend_activate,
    FILTER_INPUTS(overlay_ascend_inputs),
    FILTER_OUTPUTS(overlay_ascend_outputs),
    FILTER_QUERY_FUNC2(overlay_ascend_query_formats),
};
