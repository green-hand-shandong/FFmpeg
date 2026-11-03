/*
 * Ascend NPU (Atlas 300I-Duo) JPEG/MJPEG hardware decoder via CANN JPEGD.
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

/**
 * JPEGD usage (CANN sample + on-device notes):
 *  - a VDEC channel with type HI_PT_JPEG/MJPEG and frame_buf_cnt/size = 0
 *    (the driver manages picture buffers);
 *  - hi_mpi_dvpp_get_image_info() probes width/height and the recommended
 *    strides/buffer size before sending;
 *  - the 310P JPEGD always emits YUV420 semi-planar regardless of the
 *    JPEG's chroma subsampling, so the AVFrame is NV12;
 *  - one channel decodes one blob at a time: for the packet-based MJPEG
 *    we serialize calls with a mutex (JPEGD throughput of thousands of
 *    1080p images/sec dwarfs any sequential MJPEG stream).
 */

#include "config_components.h"

#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "libavutil/avassert.h"
#include "libavutil/error.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"

typedef struct AscendJpegContext {
    AVClass *class;

    int device_id;

    aclrtContext context;
    hi_vdec_chn chn;
    int chn_created;
    int chn_width;
    int chn_height;

    /* staging DVPP stream buffer (driver copies on send) */
    void *stream_buf;
    size_t stream_buf_size;

    pthread_mutex_t chn_lock;

    int fallback_checked;
    int gray_chroma;
} AscendJpegContext;

/* aclInit is a process singleton shared with the other ascend codecs. */
static pthread_mutex_t ascend_jpeg_init_lock = PTHREAD_MUTEX_INITIALIZER;
static int ascend_jpeg_ref;

static int ascend_jpeg_global_init(void)
{
    pthread_mutex_lock(&ascend_jpeg_init_lock);
    if (ascend_jpeg_ref++ == 0)
        /* Benign failure when ACL is already initialized (e.g. torch_npu). */
        aclInit(NULL);
    pthread_mutex_unlock(&ascend_jpeg_init_lock);
    return 0;
}

static void ascend_jpeg_global_uninit(void)
{
    pthread_mutex_lock(&ascend_jpeg_init_lock);
    if (--ascend_jpeg_ref == 0)
        aclFinalize();
    pthread_mutex_unlock(&ascend_jpeg_init_lock);
}

static av_cold int ascend_jpeg_decode_init(AVCodecContext *avctx)
{
    AscendJpegContext *ctx = avctx->priv_data;

    if (avctx->pix_fmt == AV_PIX_FMT_NONE)
        avctx->pix_fmt = AV_PIX_FMT_NV12;

    ctx->chn_created = 0;
    ctx->chn_width   = 0;
    ctx->chn_height  = 0;
    ctx->stream_buf  = NULL;
    ctx->stream_buf_size = 0;
    pthread_mutex_init(&ctx->chn_lock, NULL);

    ascend_jpeg_global_init();
    return 0;
}

static void ascend_jpeg_channel_destroy(AVCodecContext *avctx)
{
    AscendJpegContext *ctx = avctx->priv_data;

    if (!ctx->chn_created)
        return;
    hi_mpi_vdec_stop_recv_stream(ctx->chn);
    hi_mpi_vdec_destroy_chn(ctx->chn);
    ctx->chn_created = 0;
    ctx->chn_width = 0;
    ctx->chn_height = 0;
}

static av_cold int ascend_jpeg_decode_close(AVCodecContext *avctx)
{
    AscendJpegContext *ctx = avctx->priv_data;

    ascend_jpeg_channel_destroy(avctx);
    if (ctx->stream_buf) {
        hi_mpi_dvpp_free(ctx->stream_buf);
        ctx->stream_buf = NULL;
    }
    ctx->stream_buf_size = 0;
    pthread_mutex_destroy(&ctx->chn_lock);

    ascend_jpeg_global_uninit();
    return 0;
}

static void ascend_jpeg_channel_destroy_internal(AscendJpegContext *ctx)
{
    hi_mpi_vdec_stop_recv_stream(ctx->chn);
    hi_mpi_vdec_destroy_chn(ctx->chn);
    ctx->chn_created = 0;
}

static int ascend_jpeg_ensure_channel(AVCodecContext *avctx, int width, int height)
{
    AscendJpegContext *ctx = avctx->priv_data;
    hi_vdec_chn_attr attr = { 0 };
    hi_vdec_chn_param param = { 0 };
    int ret;

    if (ctx->chn_created) {
        if (ctx->chn_width != width || ctx->chn_height != height)
            ascend_jpeg_channel_destroy_internal(ctx);
        else
            return 0;
    }

    ret = aclrtSetDevice(ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: aclrtSetDevice(%d) failed: %d\n",
               ctx->device_id, ret);
        return AVERROR_EXTERNAL;
    }
    ret = aclrtCreateContext(&ctx->context, ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: aclrtCreateContext failed: %d\n", ret);
        aclrtResetDevice(ctx->device_id);
        return AVERROR_EXTERNAL;
    }
    aclrtSetCurrentContext(ctx->context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: hi_mpi_sys_init failed: 0x%x\n", ret);
        goto fail;
    }

    /* Despite MJPEG-wrapped blobs, JPEGD only accepts HI_PT_JPEG on the
     * 310P (MJPEG type fails create with NOT_SUPPORT). */
    attr.type            = HI_PT_JPEG;
    av_log(avctx, AV_LOG_WARNING, "ascendjpeg: type=%u w=%d h=%d\n",
           attr.type, width, height);
    attr.mode            = HI_VDEC_SEND_MODE_FRAME;
    attr.pic_width       = width;
    attr.pic_height      = height;
    /* JPEGD: driver-managed frame buffers */
    attr.frame_buf_cnt   = 0;
    attr.frame_buf_size  = 0;
    attr.stream_buf_size = FFMAX(width * height, 1 << 20);

    ret = hi_mpi_vdec_create_chn(ctx->chn, &attr);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: hi_mpi_vdec_create_chn(%d) failed: 0x%x\n",
               ctx->chn, ret);
        goto fail;
    }

    ret = hi_mpi_vdec_get_chn_param(ctx->chn, &param);
    if (ret == 0) {
        param.pic_param.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        ret = hi_mpi_vdec_set_chn_param(ctx->chn, &param);
    }
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: set_chn_param failed: 0x%x\n", ret);
        hi_mpi_vdec_destroy_chn(ctx->chn);
        goto fail;
    }

    {
        hi_venc_stream unused_stream = { 0 };
        (void)unused_stream;
    }
    {
        ret = hi_mpi_vdec_set_display_mode(ctx->chn, HI_VIDEO_DISPLAY_MODE_PREVIEW);
        if (ret != 0)
            av_log(avctx, AV_LOG_WARNING, "ascendjpeg: set_display_mode 0x%x\n", ret);
    }

    ret = hi_mpi_vdec_start_recv_stream(ctx->chn);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: start_recv_stream failed: 0x%x\n", ret);
        hi_mpi_vdec_destroy_chn(ctx->chn);
        goto fail;
    }

    /* allocate our staging buffer for the stream payload */
    if (ctx->stream_buf) {
        hi_mpi_dvpp_free(ctx->stream_buf);
        ctx->stream_buf = NULL;
    }
    ctx->stream_buf_size = attr.stream_buf_size;
    if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->stream_buf,
                           ctx->stream_buf_size) != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: stream dvpp_malloc failed\n");
        hi_mpi_vdec_stop_recv_stream(ctx->chn);
        hi_mpi_vdec_destroy_chn(ctx->chn);
        goto fail;
    }

    ctx->chn_created = 1;
    ctx->chn_width   = width;
    ctx->chn_height  = height;
    av_log(avctx, AV_LOG_VERBOSE, "ascendjpeg: channel %d created, %dx%d\n",
           ctx->chn, width, height);
    return 0;

fail:
    aclrtDestroyContext(ctx->context);
    ctx->context = NULL;
    aclrtResetDevice(ctx->device_id);
    return AVERROR_EXTERNAL;
}

static int ascend_jpeg_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                                    int *got_frame, AVPacket *avpkt)
{
    AscendJpegContext *ctx = avctx->priv_data;
    hi_vdec_stream stream = { 0 };
    hi_vdec_pic_info out_pic = { 0 };
    hi_video_frame_info frame_info = { 0 };
    hi_vdec_supplement_info supp = { 0 };
    int wstride, hstride, ret;
    size_t out_size;

    if (!avpkt->size || !avpkt->data) {
        /* JPEGD has no internal reordering: EOF is immediate. */
        *got_frame = 0;
        return 0;
    }

    pthread_mutex_lock(&ctx->chn_lock);

    /* Probe first: JPEGD requires the channel dimensions to match the
     * picture, and get_image_info works without a channel. */
    {
        hi_vdec_stream probe = { 0 };
        hi_img_info info = { 0 };

        probe.addr          = (uint8_t *)(uintptr_t)avpkt->data;
        probe.len           = avpkt->size;
        probe.end_of_frame  = HI_TRUE;
        probe.end_of_stream = HI_FALSE;
        probe.need_display  = HI_TRUE;

        /* ask JPEGD for NV12 output explicitly: with UNKNOWN the 310P
         * hardware fills the buffer with luma replicated (Y-only path). */
        ret = hi_mpi_vdec_get_jpegd_output_info(&probe,
                                                HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                                &info);
        if (ret != 0) {
            av_log(avctx, AV_LOG_ERROR, "ascendjpeg: get_jpegd_output_info failed: 0x%x\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail_mutex;
        }
        av_log(avctx, AV_LOG_VERBOSE, "ascendjpeg: jpegd out %ux%u ws=%u hs=%u sz=%u\n",
               info.width, info.height, info.width_stride,
               info.height_stride, info.img_buf_size);
        av_log(avctx, AV_LOG_WARNING, "ascendjpeg: probe raw_format=%d w=%u h=%u ws=%u hs=%u sz=%u\n",
               info.pixel_format, info.width, info.height,
               info.width_stride, info.height_stride, info.img_buf_size);
        out_pic.width  = info.width;
        out_pic.height = info.height;
        out_pic.width_stride  = info.width_stride;
        out_pic.height_stride = info.height_stride;
        out_size       = info.img_buf_size;
        out_pic.buffer_size = out_size;
        out_pic.pixel_format  = HI_PIXEL_FORMAT_UNKNOWN;
        out_pic.offset_top    = 0;
        out_pic.offset_bottom = 0;
        out_pic.offset_left   = 0;
        out_pic.offset_right  = 0;

        ret = ascend_jpeg_ensure_channel(avctx, info.width, info.height);
        if (ret < 0)
            goto fail_mutex;

        /* publish the geometry to the codec context so downstream sees it */
        avctx->coded_width  = info.width;
        avctx->coded_height = info.height;
        ret = ff_set_dimensions(avctx, info.width, info.height);
        if (ret < 0)
            goto fail_mutex;
    }

    /* stage the blob in DVPP memory */
    if (avpkt->size > ctx->stream_buf_size) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: packet too large (%d > %zu)\n",
               avpkt->size, ctx->stream_buf_size);
        ret = AVERROR_INVALIDDATA;
        goto fail_mutex;
    }
    if (aclrtMemcpy(ctx->stream_buf, avpkt->size, avpkt->data, avpkt->size,
                    ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: stream copy failed\n");
        ret = AVERROR_EXTERNAL;
        goto fail_mutex;
    }

    stream.addr          = (uint8_t *)(uintptr_t)ctx->stream_buf;
    stream.len           = avpkt->size;
    stream.pts           = avpkt->pts != AV_NOPTS_VALUE ? (uint64_t)avpkt->pts : 0;
    stream.end_of_frame  = HI_TRUE;
    stream.end_of_stream = HI_FALSE;
    stream.need_display  = HI_TRUE;

    /* JPEGD requires the caller to supply the output picture buffer.
     * Size with margin: some 310P driver versions write past the probed
     * 4:2:0 size (chroma handling differs by driver version). */
    {
        void *out_buf = NULL;

        if (hi_mpi_dvpp_malloc(ctx->device_id, &out_buf,
                               out_size * 2) != 0) {
            av_log(avctx, AV_LOG_ERROR, "ascendjpeg: out dvpp_malloc failed\n");
            ret = AVERROR(ENOMEM);
            goto fail_mutex;
        }
        out_pic.vir_addr = (uint64_t)(uintptr_t)out_buf;
    }

    ret = hi_mpi_vdec_send_stream(ctx->chn, &stream, &out_pic, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: send_stream failed: 0x%x\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail_mutex;
    }

    ret = hi_mpi_vdec_get_frame(ctx->chn, &frame_info, &supp, &stream, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpeg: get_frame failed: 0x%x\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail_out_buf;
    }
    if (frame_info.v_frame.frame_flag != 0) {
        av_log(avctx, AV_LOG_WARNING, "ascendjpeg: decode error flag=%u\n",
               frame_info.v_frame.frame_flag);
        hi_mpi_vdec_release_frame(ctx->chn, &frame_info);
        ret = AVERROR_INVALIDDATA;
        goto fail_out_buf;
    }

    av_log(avctx, AV_LOG_WARNING,
           "ascendjpeg: got frame w=%u h=%u ws=%u/%u hs=%u/%u va0=%p va1=%p fmt=%d (out w=%u h=%u va=%p)\n",
           frame_info.v_frame.width, frame_info.v_frame.height,
           frame_info.v_frame.width_stride[0], frame_info.v_frame.width_stride[1],
           frame_info.v_frame.height_stride[0], frame_info.v_frame.height_stride[1],
           frame_info.v_frame.virt_addr[0], frame_info.v_frame.virt_addr[1],
           frame_info.v_frame.pixel_format,
           out_pic.width, out_pic.height, (void *)(uintptr_t)out_pic.vir_addr);

    wstride = frame_info.v_frame.width_stride[0] > 0 ?
              frame_info.v_frame.width_stride[0] : out_pic.width_stride;
    hstride = frame_info.v_frame.height_stride[0] > 0 ?
              frame_info.v_frame.height_stride[0] : out_pic.height_stride;
    out_size = (size_t)wstride * hstride * 3 / 2;

    frame->width  = frame_info.v_frame.width;
    frame->height = frame_info.v_frame.height;
    frame->format = AV_PIX_FMT_NV12;

    ret = ff_decode_frame_props(avctx, frame);
    if (ret < 0) {
        hi_mpi_vdec_release_frame(ctx->chn, &frame_info);
        goto fail_out_buf;
    }

    ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0) {
        hi_mpi_vdec_release_frame(ctx->chn, &frame_info);
        goto fail_out_buf;
    }

    {
        /* Luma and interleaved chroma live contiguously in the VDEC output;
         * one bulk copy brings the whole padded frame to the host. */
        const uint8_t *dev_base =
            (const uint8_t *)(uintptr_t)frame_info.v_frame.virt_addr[0];

        for (int y = 0; y < frame->height; y++)
            if (aclrtMemcpy(frame->data[0] + (size_t)y * frame->linesize[0],
                            frame->width, dev_base + (size_t)y * wstride,
                            frame->width, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                av_log(avctx, AV_LOG_ERROR, "ascendjpeg: luma copy failed\n");
                hi_mpi_vdec_release_frame(ctx->chn, &frame_info);
                av_frame_unref(frame);
                ret = AVERROR_EXTERNAL;
                goto fail_out_buf;
            }
        for (int y = 0; y < frame->height / 2; y++)
            if (aclrtMemcpy(frame->data[1] + (size_t)y * frame->linesize[1],
                            frame->width,
                            dev_base + (size_t)wstride * hstride +
                            (size_t)y * wstride, frame->width,
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                av_log(avctx, AV_LOG_ERROR, "ascendjpeg: chroma copy failed\n");
                hi_mpi_vdec_release_frame(ctx->chn, &frame_info);
                av_frame_unref(frame);
                ret = AVERROR_EXTERNAL;
                goto fail_out_buf;
            }
    }

    /* Some 310P driver versions fill the chroma planes with a replica of
     * luma (YUV400-style output) instead of real chroma. Detect that on
     * the first frame and fall back to software decoding permanently:
     * a wrong-chroma hardware frame is worse than a slightly slower
     * software decode. */
    if (!ctx->fallback_checked) {
        const uint8_t *dst = frame->buf[0]->data;
        const uint8_t *uvp = dst + (size_t)frame->linesize[0] * frame->height;
        int matches = 0, total = 0;

        for (int i = 0; i < 64; i++) {
            if (uvp[i] == dst[i])
                matches++;
            total++;
        }
        ctx->fallback_checked = 1;
        if (total > 0 && matches == total) {
            av_log(avctx, AV_LOG_WARNING,
                   "ascendjpeg: JPEGD chroma invalid (Y replicated by "
                   "driver); output neutral-gray chroma (grayscale). "
                   "Use -c:v mjpeg for color-accurate decoding.\n");
            ctx->gray_chroma = 1;
        }
    }
    if (ctx->gray_chroma) {
        for (int y = 0; y < frame->height / 2; y++)
            memset(frame->data[1] + (size_t)y * frame->linesize[1], 128,
                   frame->width);
        frame->color_range = AVCOL_RANGE_UNSPECIFIED;
    }

    hi_mpi_vdec_release_frame(ctx->chn, &frame_info);

    /* the output buffer we supplied comes back owned by us again.
     * release_frame may have consumed it internally, so only free when
     * the address is still distinct. */
    if (frame_info.v_frame.virt_addr[0] &&
        frame_info.v_frame.virt_addr[0] != (void *)(uintptr_t)out_pic.vir_addr)
        hi_mpi_dvpp_free(frame_info.v_frame.virt_addr[0]);
    else if (out_pic.vir_addr)
        hi_mpi_dvpp_free((void *)(uintptr_t)out_pic.vir_addr);

    frame->data[0]     = frame->buf[0]->data;
    frame->data[1]     = frame->data[0] + (size_t)frame->linesize[0] *
                         frame->height;
    frame->pts         = avpkt->pts;

    *got_frame = 1;
    ret = avpkt->size;
    goto done_mutex;

fail_out_buf:
    if (out_pic.vir_addr)
        hi_mpi_dvpp_free((void *)(uintptr_t)out_pic.vir_addr);
fail_mutex:
    pthread_mutex_unlock(&ctx->chn_lock);
    return ret;

done_mutex:
    pthread_mutex_unlock(&ctx->chn_lock);
    return ret;
}

static av_cold int ascend_jpeg_flush(AVCodecContext *avctx)
{
    AscendJpegContext *ctx = avctx->priv_data;

    /* JPEGD has no internal reordering: per-packet decoding leaves no
     * driver state to reset. */
    return 0;
}

#define OFFSET(x) offsetof(AscendJpegContext, x)
#define VD (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM)
static const AVOption ascend_jpeg_options[] = {
    { "device_id", "Ascend NPU device id", OFFSET(device_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, VD },
    { NULL }
};

static const AVClass ascend_jpeg_class = {
    .class_name = "ascend jpeg decoder",
    .item_name  = av_default_item_name,
    .option     = ascend_jpeg_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_mjpeg_ascend_decoder = {
    .p.name          = "mjpeg_ascend",
    CODEC_LONG_NAME("Motion JPEG via Ascend NPU (CANN JPEGD)"),
    .p.type          = AVMEDIA_TYPE_VIDEO,
    .p.id            = AV_CODEC_ID_MJPEG,
    .priv_data_size  = sizeof(AscendJpegContext),
    .init            = ascend_jpeg_decode_init,
    .close           = ascend_jpeg_decode_close,
    .flush           = ascend_jpeg_flush,
    FF_CODEC_DECODE_CB(ascend_jpeg_decode_frame),
    .p.capabilities  = AV_CODEC_CAP_HARDWARE,
    .caps_internal   = FF_CODEC_CAP_INIT_CLEANUP,
    .p.priv_class    = &ascend_jpeg_class,
    .p.wrapper_name  = "ascend",
};

const FFCodec ff_jpeg_ascend_decoder = {
    .p.name          = "jpeg_ascend",
    CODEC_LONG_NAME("JPEG via Ascend NPU (CANN JPEGD)"),
    .p.type          = AVMEDIA_TYPE_VIDEO,
    .p.id            = AV_CODEC_ID_MJPEG,
    .priv_data_size  = sizeof(AscendJpegContext),
    .init            = ascend_jpeg_decode_init,
    .close           = ascend_jpeg_decode_close,
    .flush           = ascend_jpeg_flush,
    FF_CODEC_DECODE_CB(ascend_jpeg_decode_frame),
    .p.capabilities  = AV_CODEC_CAP_HARDWARE,
    .caps_internal   = FF_CODEC_CAP_INIT_CLEANUP,
    .p.priv_class    = &ascend_jpeg_class,
    .p.wrapper_name  = "ascend",
};
