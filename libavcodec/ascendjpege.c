/*
 * Ascend NPU (Atlas 300I-Duo) JPEG hardware encoder via CANN JPEGE.
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
 * JPEGE usage (CANN sample + on-device notes):
 *  - a VENC channel with type HI_PT_JPEG and rc_mode MJPEG-less: the
 *    jpeg_attr carries no rc; quality is set via hi_mpi_venc_set_jpeg_param
 *    (qfactor 1..100);
 *  - hi_mpi_venc_get_jpege_predicted_size() sizes the output buffer
 *    before encoding;
 *  - one channel encodes one image at a time; images are staged in a
 *    DVPP buffer and the encoded JPEG is read from the stream packs.
 */

#include "config_components.h"

#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "libavutil/error.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"

typedef struct AscendJpegeContext {
    AVClass *class;

    int device_id;
    int qfactor;

    aclrtContext context;
    hi_venc_chn chn;
    int chn_created;

    void *in_buf;
    size_t in_buf_size;
    void *stream_buf;
    size_t stream_buf_size;

    pthread_mutex_t chn_lock;
} AscendJpegeContext;

static pthread_mutex_t ascend_jpege_init_lock = PTHREAD_MUTEX_INITIALIZER;
static int ascend_jpege_ref;

static int ascend_jpege_global_init(void)
{
    pthread_mutex_lock(&ascend_jpege_init_lock);
    if (ascend_jpege_ref++ == 0)
        aclInit(NULL);
    pthread_mutex_unlock(&ascend_jpege_init_lock);
    return 0;
}

static void ascend_jpege_global_uninit(void)
{
    pthread_mutex_lock(&ascend_jpege_init_lock);
    if (--ascend_jpege_ref == 0)
        aclFinalize();
    pthread_mutex_unlock(&ascend_jpege_init_lock);
}

static av_cold int ascend_jpege_init(AVCodecContext *avctx)
{
    AscendJpegeContext *ctx = avctx->priv_data;

    if (avctx->pix_fmt != AV_PIX_FMT_NV12) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: only NV12 input is supported\n");
        return AVERROR(EINVAL);
    }

    pthread_mutex_init(&ctx->chn_lock, NULL);
    ascend_jpege_global_init();
    return 0;
}

static void ascend_jpege_channel_destroy(AscendJpegeContext *ctx)
{
    if (!ctx->chn_created)
        return;
    hi_mpi_venc_stop_chn(ctx->chn);
    hi_mpi_venc_destroy_chn(ctx->chn);
    ctx->chn_created = 0;
}

static av_cold int ascend_jpege_close(AVCodecContext *avctx)
{
    AscendJpegeContext *ctx = avctx->priv_data;

    ascend_jpege_channel_destroy(ctx);
    if (ctx->in_buf) {
        hi_mpi_dvpp_free(ctx->in_buf);
        ctx->in_buf = NULL;
    }
    if (ctx->stream_buf) {
        hi_mpi_dvpp_free(ctx->stream_buf);
        ctx->stream_buf = NULL;
    }
    if (ctx->context) {
        aclrtDestroyContext(ctx->context);
        ctx->context = NULL;
    }
    aclrtResetDevice(ctx->device_id);
    pthread_mutex_destroy(&ctx->chn_lock);
    ascend_jpege_global_uninit();
    return 0;
}

static int ascend_jpege_ensure_channel(AVCodecContext *avctx, int width, int height)
{
    AscendJpegeContext *ctx = avctx->priv_data;
    hi_venc_chn_attr attr = { 0 };
    hi_venc_start_param start = { 0 };
    hi_venc_jpeg_param jpeg_param = { 0 };
    int wstride = FFALIGN(width, 16);
    int hstride = FFALIGN(height, 2);
    int ret;

    if (ctx->chn_created)
        return 0;

    ret = aclrtSetDevice(ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: aclrtSetDevice failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }
    ret = aclrtCreateContext(&ctx->context, ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: aclrtCreateContext failed: %d\n", ret);
        aclrtResetDevice(ctx->device_id);
        return AVERROR_EXTERNAL;
    }
    aclrtSetCurrentContext(ctx->context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: sys_init failed: 0x%x\n", ret);
        goto fail;
    }

    attr.venc_attr.type           = HI_PT_JPEG;
    attr.venc_attr.max_pic_width  = wstride;
    attr.venc_attr.max_pic_height = hstride;
    attr.venc_attr.pic_width      = width;
    attr.venc_attr.pic_height     = height;
    attr.venc_attr.buf_size       = FFALIGN((unsigned)width * (unsigned)width, 64);
    attr.venc_attr.profile        = 0; /* Baseline */
    attr.venc_attr.is_by_frame    = HI_TRUE;
    attr.venc_attr.jpeg_attr.dcf_en    = HI_FALSE;
    attr.venc_attr.jpeg_attr.recv_mode = HI_VENC_PIC_RECV_SINGLE;
    attr.venc_attr.jpeg_attr.mpf_cfg.large_thumbnail_num = 0;

    ret = hi_mpi_venc_create_chn(ctx->chn, &attr);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: create_chn failed: 0x%x\n", ret);
        goto fail;
    }

    ret = hi_mpi_venc_get_jpeg_param(ctx->chn, &jpeg_param);
    if (ret == 0) {
        jpeg_param.qfactor = FFMIN(100, FFMAX(1, ctx->qfactor));
        ret = hi_mpi_venc_set_jpeg_param(ctx->chn, &jpeg_param);
    }
    if (ret != 0)
        av_log(avctx, AV_LOG_WARNING, "ascendjpege: set_jpeg_param failed: 0x%x\n", ret);

    start.recv_pic_num = -1;
    ret = hi_mpi_venc_start_chn(ctx->chn, &start);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: start_chn failed: 0x%x\n", ret);
        hi_mpi_venc_destroy_chn(ctx->chn);
        goto fail;
    }

    ctx->in_buf_size = (size_t)wstride * hstride * 3 / 2;
    if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->in_buf, ctx->in_buf_size) != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: in dvpp_malloc failed\n");
        hi_mpi_venc_stop_chn(ctx->chn);
        hi_mpi_venc_destroy_chn(ctx->chn);
        goto fail;
    }
    ctx->stream_buf_size = FFMAX(1 << 20, ctx->in_buf_size);
    if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->stream_buf, ctx->stream_buf_size) != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: stream dvpp_malloc failed\n");
        hi_mpi_dvpp_free(ctx->in_buf);
        ctx->in_buf = NULL;
        hi_mpi_venc_stop_chn(ctx->chn);
        hi_mpi_venc_destroy_chn(ctx->chn);
        goto fail;
    }

    ctx->chn_created = 1;
    return 0;

fail:
    aclrtDestroyContext(ctx->context);
    ctx->context = NULL;
    aclrtResetDevice(ctx->device_id);
    return AVERROR_EXTERNAL;
}

static int ascend_jpege_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                                     const AVFrame *frame, int *got_packet)
{
    AscendJpegeContext *ctx = avctx->priv_data;
    hi_video_frame_info fi = { 0 };
    hi_venc_stream stream = { 0 };
    hi_venc_chn_status st = { 0 };
    size_t total = 0;
    int wstride, hstride, ret;

    pthread_mutex_lock(&ctx->chn_lock);

    wstride = FFALIGN(frame->width, 16);
    hstride = FFALIGN(frame->height, 2);

    ret = ascend_jpege_ensure_channel(avctx, frame->width, frame->height);
    if (ret < 0)
        goto fail_mutex;

    aclrtSetCurrentContext(ctx->context);

    /* stage NV12 into DVPP memory */
    {
        uint8_t *dst = ctx->in_buf;

        for (int y = 0; y < frame->height; y++)
            if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                            frame->data[0] + (size_t)y * frame->linesize[0],
                            frame->width, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
                ret = AVERROR_EXTERNAL;
                goto fail_mutex;
            }
        dst += (size_t)wstride * hstride;
        for (int y = 0; y < (frame->height + 1) / 2; y++)
            if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                            frame->data[1] + (size_t)y * frame->linesize[1],
                            frame->width, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
                ret = AVERROR_EXTERNAL;
                goto fail_mutex;
            }
    }

    fi.v_frame.width            = frame->width;
    fi.v_frame.height           = frame->height;
    fi.v_frame.pixel_format     = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    fi.v_frame.video_format     = HI_VIDEO_FORMAT_LINEAR;
    fi.v_frame.compress_mode    = HI_COMPRESS_MODE_NONE;
    fi.v_frame.dynamic_range    = HI_DYNAMIC_RANGE_SDR8;
    fi.v_frame.field            = HI_VIDEO_FIELD_FRAME;
    fi.v_frame.color_gamut      = HI_COLOR_GAMUT_BT709;
    fi.v_frame.width_stride[0]  = wstride;
    fi.v_frame.width_stride[1]  = wstride;
    fi.v_frame.virt_addr[0]     = ctx->in_buf;
    fi.v_frame.virt_addr[1]     = (void *)((uintptr_t)ctx->in_buf +
                                  (size_t)wstride * hstride);
    fi.v_frame.header_stride[0] = 0;
    fi.v_frame.header_stride[1] = 0;
    fi.v_frame.header_virt_addr[0] = fi.v_frame.virt_addr[0];
    fi.v_frame.header_virt_addr[1] = fi.v_frame.virt_addr[0];
    fi.v_frame.pts              = 0;
    fi.pool_id                  = 0;

    fprintf(stderr, "[jpege] send: va0=%p va1=%p ws=%u/%u w=%d h=%d fmt=%d vf=%d cmp=%d dr=%d field=%d gamut=%d pool=%d\n",
            fi.v_frame.virt_addr[0], fi.v_frame.virt_addr[1],
            fi.v_frame.width_stride[0], fi.v_frame.width_stride[1],
            fi.v_frame.width, fi.v_frame.height, fi.v_frame.pixel_format,
            fi.v_frame.video_format, fi.v_frame.compress_mode,
            fi.v_frame.dynamic_range, fi.v_frame.field, fi.v_frame.color_gamut,
            fi.pool_id);
    ret = hi_mpi_venc_send_frame(ctx->chn, &fi, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: send_frame failed: 0x%x\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail_mutex;
    }

    stream.pack = av_malloc_array(2, sizeof(*stream.pack));
    if (!stream.pack) {
        ret = AVERROR(ENOMEM);
        goto fail_mutex;
    }
    stream.pack_cnt = 2;

    ret = hi_mpi_venc_get_stream(ctx->chn, &stream, 1000);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendjpege: get_stream failed: 0x%x\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail_mutex;
    }

    for (uint32_t i = 0; i < stream.pack_cnt; i++)
        total += stream.pack[i].len;

    ret = ff_get_encode_buffer(avctx, pkt, FFMAX(total, 1), 0);
    if (ret < 0) {
        hi_mpi_venc_release_stream(ctx->chn, &stream);
        av_freep(&stream.pack);
        goto fail_mutex;
    }
    pkt->size = 0;
    for (uint32_t i = 0; i < stream.pack_cnt; i++) {
        if (stream.pack[i].addr && stream.pack[i].len) {
            if (aclrtMemcpy(pkt->data + pkt->size, stream.pack[i].len,
                            stream.pack[i].addr, stream.pack[i].len,
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                hi_mpi_venc_release_stream(ctx->chn, &stream);
                av_freep(&stream.pack);
                ret = AVERROR_EXTERNAL;
                goto fail_mutex;
            }
            pkt->size += stream.pack[i].len;
        }
    }
    pkt->flags |= AV_PKT_FLAG_KEY;

    hi_mpi_venc_release_stream(ctx->chn, &stream);
    av_freep(&stream.pack);

    *got_packet = 1;
    ret = 0;
    goto done_mutex;

fail_mutex:
    pthread_mutex_unlock(&ctx->chn_lock);
    return ret;

done_mutex:
    pthread_mutex_unlock(&ctx->chn_lock);
    return ret;
}

#define OFFSET(x) offsetof(AscendJpegeContext, x)
#define VE (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption ascend_jpege_options[] = {
    { "device_id", "Ascend NPU device id", OFFSET(device_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, VE },
    { "qfactor", "JPEG quality factor (1..100)", OFFSET(qfactor),
      AV_OPT_TYPE_INT, { .i64 = 90 }, 1, 100, VE },
    { NULL }
};

static const AVClass ascend_jpege_class = {
    .class_name = "ascend jpege encoder",
    .item_name  = av_default_item_name,
    .option     = ascend_jpege_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_jpeg_ascend_encoder = {
    .p.name          = "jpeg_ascend",
    CODEC_LONG_NAME("JPEG via Ascend NPU (CANN JPEGE)"),
    .p.type          = AVMEDIA_TYPE_VIDEO,
    .p.id            = AV_CODEC_ID_MJPEG,
    .priv_data_size  = sizeof(AscendJpegeContext),
    .init            = ascend_jpege_init,
    FF_CODEC_ENCODE_CB(ascend_jpege_encode_frame),
    .close           = ascend_jpege_close,
    .p.capabilities  = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_DR1,
    .caps_internal   = FF_CODEC_CAP_INIT_CLEANUP,
    .p.priv_class    = &ascend_jpege_class,
    .p.wrapper_name  = "ascend",
    CODEC_PIXFMTS(AV_PIX_FMT_NV12),
};
