/*
 * Ascend NPU (Atlas 300I-Duo / Ascend 310P) hardware video encoder
 * via the CANN DVPP media APIs (hi_mpi_venc).
 *
 * NOTE for CLI users: the hardware encoder is much faster than real-time;
 * when transcoding offline files the CLI's default fps-mode may drop
 * frames to keep sync. Pass -fps_mode passthrough for file transcodes
 * (the standard practice for faster-than-realtime encoders).
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

#include "config_components.h"

#include <pthread.h>
#include <unistd.h>
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

#define ASCEND_MAX_CHN 32
#define ASCEND_ENC_TIMEOUT_MS 1000

static pthread_mutex_t ascend_enc_global_lock = PTHREAD_MUTEX_INITIALIZER;
static int ascend_enc_global_ref;
static uint32_t ascend_enc_chn_used[(ASCEND_MAX_CHN + 31) / 32];

typedef struct AscendEncContext {
    AVClass *class;

    int device_id;
    int rc_mode;      /* 0=cbr, 1=vbr, 2=avbr, 3=fixqp */
    int gop;
    int qp;           /* fixqp I qp; P = qp + 2 */
    int zero_copy;    /* feed hw frames to VENC without staging copy */

    int chn_created;
    hi_venc_chn chn;
    aclrtContext context;
    int pending_frames;      /* frames sent, packets not yet collected */

    uint64_t frame_count;
    uint64_t frame_count_out;
    int64_t  pts_lut[64];      /* submitted frame->pts ring, consumed FIFO */
    int      pts_lut_count;

    /* staging DVPP input buffers; VENC references them asynchronously,
     * so frames are staged in rotation. */
    void *in_buf[4];
    int in_buf_count;

    /* zero-copy mode: pending input frame buf refs held until the
     * corresponding encoded packet is collected */
    AVBufferRef *pending_bufs[16];
    int pending_buf_count;
    AVPacket *out_q[16];
    int out_q_count;
    int in_buf_pos;
    size_t in_buf_size;
} AscendEncContext;

static int ascend_enc_global_init(void)
{
    pthread_mutex_lock(&ascend_enc_global_lock);
    if (ascend_enc_global_ref++ == 0)
        /* Benign failure when the process already initialized ACL. */
        aclInit(NULL);
    pthread_mutex_unlock(&ascend_enc_global_lock);
    return 0;
}

static void ascend_enc_global_uninit(int device_id)
{
    pthread_mutex_lock(&ascend_enc_global_lock);
    if (--ascend_enc_global_ref == 0) {
        aclrtResetDevice(device_id);
        aclFinalize();
    }
    pthread_mutex_unlock(&ascend_enc_global_lock);
}

static int ascend_enc_chn_alloc(hi_venc_chn *chn)
{
    int ret = AVERROR(ENOSYS);

    pthread_mutex_lock(&ascend_enc_global_lock);
    for (int i = 0; i < ASCEND_MAX_CHN; i++) {
        if (!(ascend_enc_chn_used[i / 32] & (1u << (i % 32)))) {
            ascend_enc_chn_used[i / 32] |= 1u << (i % 32);
            *chn = i;
            ret = 0;
            break;
        }
    }
    pthread_mutex_unlock(&ascend_enc_global_lock);
    return ret;
}

static void ascend_enc_chn_free(hi_venc_chn chn)
{
    pthread_mutex_lock(&ascend_enc_global_lock);
    ascend_enc_chn_used[chn / 32] &= ~(1u << (chn % 32));
    pthread_mutex_unlock(&ascend_enc_global_lock);
}

/* Pick the first usable Ascend device by probing the runtime's logical
 * device ids (container-aware). */
static int ascend_enc_pick_device(AVCodecContext *avctx, AscendEncContext *ctx)
{
    unsigned int count = 0;
    unsigned int id;

    if (aclrtGetDeviceCount(&count) != 0 || count == 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: no Ascend devices present\n");
        return AVERROR(ENODEV);
    }

    for (id = 0; id < count; id++) {
        ctx->device_id = (int)id;
        av_log(avctx, AV_LOG_VERBOSE, "ascendenc: auto-picked device %u\n", id);
        return 0;
    }
    return AVERROR(ENODEV);
}

static av_cold int ascend_encode_init(AVCodecContext *avctx)
{
    AscendEncContext *ctx = avctx->priv_data;
    hi_payload_type type = avctx->codec_id == AV_CODEC_ID_HEVC ?
                           HI_PT_H265 : HI_PT_H264;
    hi_venc_chn_attr attr = { 0 };
    hi_venc_start_param start = { 0 };
    int wstride, hstride, fps, ret;

    if (avctx->pix_fmt != AV_PIX_FMT_NV12 &&
        avctx->pix_fmt != AV_PIX_FMT_ASCEND) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: only NV12 input is supported\n");
        return AVERROR(EINVAL);
    }
    if (avctx->bit_rate <= 0) {
        avctx->bit_rate = 4 * 1000 * 1000;
        av_log(avctx, AV_LOG_WARNING, "ascendenc: no bitrate set, using %d kbps\n",
               avctx->bit_rate / 1000);
    }

    if (ctx->device_id < 0) {
        int ret = ascend_enc_pick_device(avctx, ctx);
        if (ret < 0)
            return ret;
    }

    fps = avctx->framerate.num > 0 ?
          avctx->framerate.num / FFMAX(avctx->framerate.den, 1) : 25;
    fps = FFMAX(fps, 1);

    /* VENC input layout (matching the CANN sample): width stride 32-aligned
     * bytes, height 2-aligned. */
    wstride = FFALIGN(avctx->width, 32);
    hstride = FFALIGN(avctx->height, 2);
    ctx->in_buf_size = (size_t)wstride * hstride * 3 / 2;

    ret = aclrtSetDevice(ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: aclrtSetDevice(%d) failed: %d\n",
               ctx->device_id, ret);
        return AVERROR_EXTERNAL;
    }
    ret = aclrtCreateContext(&ctx->context, ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: aclrtCreateContext failed: %d\n", ret);
        aclrtResetDevice(ctx->device_id);
        return AVERROR_EXTERNAL;
    }
    aclrtSetCurrentContext(ctx->context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: hi_mpi_sys_init failed: 0x%x\n", ret);
        goto fail_device;
    }

    ret = ascend_enc_chn_alloc(&ctx->chn);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: no free VENC channel\n");
        goto fail_device;
    }

    attr.venc_attr.type           = type;
    attr.venc_attr.max_pic_width  = FFALIGN(avctx->width, 32);
    attr.venc_attr.max_pic_height = FFALIGN(avctx->height, 2);
    attr.venc_attr.buf_size       = 1024 * 1024 * 4; /* stream buffer, sample value */
    attr.venc_attr.profile        = type == HI_PT_H265 ? 0 : 2; /* MP / HP */
    attr.venc_attr.is_by_frame    = HI_TRUE;
    attr.venc_attr.pic_width      = avctx->width;
    attr.venc_attr.pic_height     = avctx->height;
    attr.venc_attr.h264_attr.rcn_ref_share_buf_en = HI_FALSE;
    attr.venc_attr.h264_attr.frame_buf_ratio      = 0;

    {
        const uint32_t gop = ctx->gop > 0 ? ctx->gop : 250;

        attr.gop_attr.gop_mode = HI_VENC_GOP_MODE_NORMAL_P;
        attr.gop_attr.normal_p.ip_qp_delta = 2;

        switch (ctx->rc_mode) {
        case 1: /* vbr */
            attr.rc_attr.rc_mode = type == HI_PT_H265 ?
                HI_VENC_RC_MODE_H265_VBR : HI_VENC_RC_MODE_H264_VBR;
            attr.rc_attr.h264_vbr.gop            = gop;
            attr.rc_attr.h264_vbr.stats_time     = 1;
            attr.rc_attr.h264_vbr.src_frame_rate = fps;
            attr.rc_attr.h264_vbr.dst_frame_rate = fps;
            attr.rc_attr.h264_vbr.max_bit_rate   = avctx->bit_rate / 1000;
            break;
        case 2: /* avbr */
            attr.rc_attr.rc_mode = type == HI_PT_H265 ?
                HI_VENC_RC_MODE_H265_AVBR : HI_VENC_RC_MODE_H264_AVBR;
            attr.rc_attr.h264_avbr.gop            = gop;
            attr.rc_attr.h264_avbr.stats_time     = 1;
            attr.rc_attr.h264_avbr.src_frame_rate = fps;
            attr.rc_attr.h264_avbr.dst_frame_rate = fps;
            attr.rc_attr.h264_avbr.max_bit_rate = avctx->bit_rate / 1000;
            break;
        case 3: /* fixqp */
            attr.rc_attr.rc_mode = type == HI_PT_H265 ?
                HI_VENC_RC_MODE_H265_FIXQP : HI_VENC_RC_MODE_H264_FIXQP;
            attr.rc_attr.h264_fixqp.gop            = gop;
            attr.rc_attr.h264_fixqp.src_frame_rate = fps;
            attr.rc_attr.h264_fixqp.dst_frame_rate = fps;
            attr.rc_attr.h264_fixqp.i_qp           = FFMIN(51, FFMAX(0, ctx->qp));
            attr.rc_attr.h264_fixqp.p_qp           = FFMIN(51, FFMAX(0, ctx->qp + 2));
            attr.rc_attr.h264_fixqp.b_qp           = FFMIN(51, FFMAX(0, ctx->qp + 4));
            break;
        case 0: /* cbr */
        default:
            attr.rc_attr.rc_mode = type == HI_PT_H265 ?
                HI_VENC_RC_MODE_H265_CBR : HI_VENC_RC_MODE_H264_CBR;
            attr.rc_attr.h264_cbr.gop            = gop;
            attr.rc_attr.h264_cbr.stats_time     = 1;
            attr.rc_attr.h264_cbr.src_frame_rate = fps;
            attr.rc_attr.h264_cbr.dst_frame_rate = fps;
            attr.rc_attr.h264_cbr.bit_rate       = avctx->bit_rate / 1000;
            break;
        }
    }

    ret = hi_mpi_venc_create_chn(ctx->chn, &attr);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR,
               "ascendenc: hi_mpi_venc_create_chn(%d) failed: 0x%x "
               "(%dx%d max=%ux%u buf=%u prof=%u byframe=%d rc=%u gop=%u br=%u)\n",
               ctx->chn, ret, attr.venc_attr.pic_width, attr.venc_attr.pic_height,
               attr.venc_attr.max_pic_width, attr.venc_attr.max_pic_height,
               attr.venc_attr.buf_size, attr.venc_attr.profile,
               attr.venc_attr.is_by_frame, attr.rc_attr.rc_mode,
               attr.rc_attr.h264_cbr.gop, attr.rc_attr.h264_cbr.bit_rate);
        ascend_enc_chn_free(ctx->chn);
        goto fail_device;
    }

    start.recv_pic_num = -1;
    ret = hi_mpi_venc_start_chn(ctx->chn, &start);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascendenc: hi_mpi_venc_start_chn failed: 0x%x\n", ret);
        hi_mpi_venc_destroy_chn(ctx->chn);
        ascend_enc_chn_free(ctx->chn);
        goto fail_device;
    }

    for (int i = 0; i < FF_ARRAY_ELEMS(ctx->in_buf); i++) {
        if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->in_buf[i],
                               ctx->in_buf_size) != 0) {
            av_log(avctx, AV_LOG_ERROR, "ascendenc: input dvpp_malloc failed\n");
            ctx->in_buf_count = i;
            hi_mpi_venc_stop_chn(ctx->chn);
            hi_mpi_venc_destroy_chn(ctx->chn);
            ascend_enc_chn_free(ctx->chn);
            goto fail_device;
        }
    }
    ctx->in_buf_count = FF_ARRAY_ELEMS(ctx->in_buf);

    ctx->chn_created = 1;
    av_log(avctx, AV_LOG_VERBOSE, "ascendenc: channel %d created %dx%d@%dkbps rc=%d\n",
           ctx->chn, avctx->width, avctx->height, avctx->bit_rate / 1000,
           ctx->rc_mode);
    return 0;

fail_device:
    aclrtDestroyContext(ctx->context);
    ctx->context = NULL;
    aclrtResetDevice(ctx->device_id);
    return AVERROR_EXTERNAL;
}

/* Upload a software NV12 frame into the DVPP staging buffer. */
static int ascend_upload_frame(AVCodecContext *avctx, const AVFrame *frame)
{
    AscendEncContext *ctx = avctx->priv_data;
    const int wstride = FFALIGN(avctx->width, 32);
    const int hstride = FFALIGN(avctx->height, 2);
    uint8_t *dst = ctx->in_buf[ctx->in_buf_pos];

    for (int y = 0; y < frame->height; y++)
        if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                        frame->data[0] + (size_t)y * frame->linesize[0],
                        frame->width, ACL_MEMCPY_HOST_TO_DEVICE) != 0)
            return AVERROR_EXTERNAL;
    dst += (size_t)wstride * hstride;
    for (int y = 0; y < (frame->height + 1) / 2; y++)
        if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                        frame->data[1] + (size_t)y * frame->linesize[1],
                        frame->width, ACL_MEMCPY_HOST_TO_DEVICE) != 0)
            return AVERROR_EXTERNAL;
    return 0;
}

/* Fetch one encoded access unit from VENC into pkt.
 * Returns 0 with *got_packet_ptr set, or a negative error. */
static int ascend_collect_stream(AVCodecContext *avctx, AVPacket *pkt,
                                 int timeout_ms, int *got_packet_ptr)
{
    AscendEncContext *ctx = avctx->priv_data;
    hi_venc_stream stream = { 0 };
    size_t total = 0;
    int ret;

    aclrtSetCurrentContext(ctx->context);

    stream.pack = av_malloc_array(2, sizeof(*stream.pack));
    if (!stream.pack)
        return AVERROR(ENOMEM);
    stream.pack_cnt = 2;

    ret = hi_mpi_venc_get_stream(ctx->chn, &stream, timeout_ms);
    if (ret != 0) {
        av_freep(&stream.pack);
        *got_packet_ptr = 0;
        return 0;
    }

    for (uint32_t i = 0; i < stream.pack_cnt; i++)
        total += stream.pack[i].len;
    if (total == 0) {
        hi_mpi_venc_release_stream(ctx->chn, &stream);
        av_freep(&stream.pack);
        *got_packet_ptr = 0;
        return 0;
    }

    ret = ff_get_encode_buffer(avctx, pkt, total, 0);
    if (ret < 0) {
        hi_mpi_venc_release_stream(ctx->chn, &stream);
        av_freep(&stream.pack);
        return ret;
    }
    pkt->size = 0;
    for (uint32_t i = 0; i < stream.pack_cnt; i++) {
        if (stream.pack[i].addr && stream.pack[i].len) {
            if (aclrtMemcpy(pkt->data + pkt->size, stream.pack[i].len,
                            stream.pack[i].addr, stream.pack[i].len,
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                hi_mpi_venc_release_stream(ctx->chn, &stream);
                av_freep(&stream.pack);
                return AVERROR_EXTERNAL;
            }
            pkt->size += stream.pack[i].len;
        }
    }
    /* VENC echoes the frame index we submitted; map it back to the
     * original frame pts (in the CLI-provided encoder timebase). */
    {
        int64_t orig = ctx->pts_lut_count > 0 ? ctx->pts_lut[0] :
                       (int64_t)ctx->frame_count_out;
        if (ctx->pts_lut_count > 0) {
            memmove(ctx->pts_lut, ctx->pts_lut + 1,
                    sizeof(ctx->pts_lut[0]) * (ctx->pts_lut_count - 1));
            ctx->pts_lut_count--;
        }
        pkt->pts = pkt->dts = orig;
    }
    ctx->frame_count_out++;
    {
        const hi_venc_h264_nalu_type h264t = stream.pack[0].data_type.h264_type;
        const hi_venc_h265_nalu_type h265t = stream.pack[0].data_type.h265_type;
        if (h264t == HI_VENC_H264_NALU_IDR_SLICE ||
            h264t == HI_VENC_H264_NALU_SPS ||
            h265t == HI_VENC_H265_NALU_IDR_SLICE ||
            h265t == HI_VENC_H265_NALU_SPS)
            pkt->flags |= AV_PKT_FLAG_KEY;
    }

    hi_mpi_venc_release_stream(ctx->chn, &stream);
    av_freep(&stream.pack);
    *got_packet_ptr = 1;
    ctx->pending_frames = FFMAX(ctx->pending_frames - 1, 0);
    if (ctx->pending_buf_count > 0) {
        av_buffer_unref(&ctx->pending_bufs[0]);
        memmove(&ctx->pending_bufs[0], &ctx->pending_bufs[1],
                sizeof(ctx->pending_bufs[0]) * (ctx->pending_buf_count - 1));
        ctx->pending_buf_count--;
    }
    return 0;
}

static int ascend_encode_cb(AVCodecContext *avctx, AVPacket *pkt,
                            const AVFrame *frame, int *got_packet_ptr)
{
    AscendEncContext *ctx = avctx->priv_data;
    hi_video_frame_info fi = { 0 };
    hi_venc_stream stream = { 0 };
    const int wstride = FFALIGN(avctx->width, 32);
    const int hstride = FFALIGN(avctx->height, 2);
    size_t total = 0;
    int ret;

    if (frame) {
        /* Bind the ACL context: the encoder may run on a worker thread. */
        aclrtSetCurrentContext(ctx->context);

        /* VENC input queue is shallow: drain any pending output first so
         * the send below never hits a full queue (0xa0088003). */
        if (ctx->pending_frames > 0) {
            ret = ascend_collect_stream(avctx, pkt, 0, got_packet_ptr);
            if (ret < 0)
                return ret;
            if (*got_packet_ptr)
                return 0;
        }

        if (frame->format == AV_PIX_FMT_ASCEND && ctx->zero_copy &&
            frame->buf[0]) {
            /* Zero-copy: the frame's DVPP buffer is referenced directly
             * by VENC; hold the buf ref until the encoded packet for
             * this frame is collected. */
            if (ctx->pending_buf_count >= (int)FF_ARRAY_ELEMS(ctx->pending_bufs)) {
                /* pool pressure: drain one encoded packet into the output
                 * queue (releases its input buf ref) before staging. */
                AVPacket *drain = av_packet_alloc();
                int got = 0;

                if (drain) {
                    int dr = ascend_collect_stream(avctx, drain, 1000, &got);
                    if (dr < 0) {
                        av_packet_free(&drain);
                        return dr;
                    }
                    if (got && ctx->out_q_count < (int)FF_ARRAY_ELEMS(ctx->out_q))
                        ctx->out_q[ctx->out_q_count++] = drain;
                    else
                        av_packet_free(&drain);
                }
            }
            if (ctx->pending_buf_count < (int)FF_ARRAY_ELEMS(ctx->pending_bufs)) {
                ctx->pending_bufs[ctx->pending_buf_count++] =
                    av_buffer_ref(frame->buf[0]);
            } else {
                av_log(avctx, AV_LOG_ERROR, "ascendenc: pending bufs full\n");
                return AVERROR(ENOMEM);
            }
            fi.v_frame.virt_addr[0]     = frame->data[0];
            fi.v_frame.virt_addr[1]     = frame->data[1];
            fi.v_frame.width_stride[0]  = frame->linesize[0] > 0 ?
                                          frame->linesize[0] : wstride;
            fi.v_frame.width_stride[1]  = fi.v_frame.width_stride[0];
        } else if (frame->format == AV_PIX_FMT_ASCEND) {
            /* Copy device-to-device into the staging pool: the source
             * frame's DVPP buffer returns to its producer's pool as soon
             * as this function returns, while VENC still references it. */
            const int src_stride = frame->linesize[0] > 0 ?
                                   frame->linesize[0] : wstride;
            const uint8_t *src = frame->data[0];
            uint8_t *dst = ctx->in_buf[ctx->in_buf_pos];

            for (int y = 0; y < frame->height; y++)
                if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                                src + (size_t)y * src_stride, frame->width,
                                ACL_MEMCPY_DEVICE_TO_DEVICE) != 0)
                    return AVERROR_EXTERNAL;
            dst += (size_t)wstride * hstride;
            for (int y = 0; y < (frame->height + 1) / 2; y++)
                if (aclrtMemcpy(dst + (size_t)y * wstride, frame->width,
                                frame->data[1] + (size_t)y * src_stride,
                                frame->width,
                                ACL_MEMCPY_DEVICE_TO_DEVICE) != 0)
                    return AVERROR_EXTERNAL;
            fi.v_frame.virt_addr[0]     = ctx->in_buf[ctx->in_buf_pos];
            fi.v_frame.virt_addr[1]     = (void *)((uintptr_t)ctx->in_buf[ctx->in_buf_pos] +
                                          (size_t)wstride * hstride);
            fi.v_frame.width_stride[0]  = wstride;
            fi.v_frame.width_stride[1]  = wstride;
        } else {
            ret = ascend_upload_frame(avctx, frame);
            if (ret < 0)
                return ret;
            fi.v_frame.virt_addr[0]     = ctx->in_buf[ctx->in_buf_pos];
            fi.v_frame.virt_addr[1]     = (void *)((uintptr_t)ctx->in_buf[ctx->in_buf_pos] +
                                          (size_t)wstride * hstride);
            fi.v_frame.width_stride[0]  = wstride;
            fi.v_frame.width_stride[1]  = wstride;
        }
        fi.v_frame.width            = frame->width;
        fi.v_frame.height           = frame->height;
        fi.v_frame.pixel_format     = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        fi.v_frame.video_format     = HI_VIDEO_FORMAT_LINEAR;
        fi.v_frame.compress_mode    = HI_COMPRESS_MODE_NONE;
        fi.v_frame.dynamic_range    = HI_DYNAMIC_RANGE_SDR8;
        fi.v_frame.field            = HI_VIDEO_FIELD_FRAME;
        fi.v_frame.color_gamut      = HI_COLOR_GAMUT_BT709;
        fi.pool_id                  = 0;
        fi.mod_id                   = HI_ID_VENC;
        /* time_ref must be monotonically even; VDEC/VENC pairing uses it
         * for reference marking. */
        fi.v_frame.time_ref         = (uint32_t)(ctx->frame_count * 2);
        fi.v_frame.pts              = ctx->frame_count;  /* VENC echoes this back */
        if (ctx->pts_lut_count < 64)
            ctx->pts_lut[ctx->pts_lut_count++] =
                frame->pts != AV_NOPTS_VALUE ? frame->pts : (int64_t)ctx->frame_count;

        ret = hi_mpi_venc_send_frame(ctx->chn, &fi, ASCEND_ENC_TIMEOUT_MS);
        if (ret != 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "ascendenc: send_frame failed: 0x%x (w=%u h=%u ws=%u/%u "
                   "va0=%p va1=%p pts=%llu)\n",
                   ret, fi.v_frame.width, fi.v_frame.height,
                   fi.v_frame.width_stride[0], fi.v_frame.width_stride[1],
                   fi.v_frame.virt_addr[0], fi.v_frame.virt_addr[1],
                   (unsigned long long)fi.v_frame.pts);
            return AVERROR_EXTERNAL;
        }
        ctx->frame_count++;
        ctx->pending_frames++;
        ctx->in_buf_pos = (ctx->in_buf_pos + 1) % ctx->in_buf_count;

        /* zero-copy staging holds pool buffers: force a synchronized
         * collect so pending refs don't exhaust upstream pools. */
        if (ctx->zero_copy && frame->format == AV_PIX_FMT_ASCEND) {
            ret = ascend_collect_stream(avctx, pkt, 200, got_packet_ptr);
            if (ret < 0)
                return ret;
            if (*got_packet_ptr)
                return 0;
        }
    }

    /* First emit any packet queued during backpressure. */
    if (ctx->out_q_count > 0) {
        av_packet_move_ref(pkt, ctx->out_q[0]);
        av_packet_free(&ctx->out_q[0]);
        memmove(&ctx->out_q[0], &ctx->out_q[1],
                sizeof(ctx->out_q[0]) * (ctx->out_q_count - 1));
        ctx->out_q_count--;
        *got_packet_ptr = 1;
        return 0;
    }

    /* Collect whatever encoded output is ready. */
    ret = ascend_collect_stream(avctx, pkt, frame ? 0 : ASCEND_ENC_TIMEOUT_MS,
                                got_packet_ptr);
    if (ret == 0 && !*got_packet_ptr && !frame)
        return AVERROR_EOF; /* flush request and nothing left */
    return ret;
}

static av_cold int ascend_encode_close(AVCodecContext *avctx)
{
    AscendEncContext *ctx = avctx->priv_data;

    if (ctx->chn_created) {
        hi_mpi_venc_stop_chn(ctx->chn);
        hi_mpi_venc_destroy_chn(ctx->chn);
        ascend_enc_chn_free(ctx->chn);
        ctx->chn_created = 0;
    }
    for (int i = 0; i < ctx->in_buf_count; i++)
        if (ctx->in_buf[i])
            hi_mpi_dvpp_free(ctx->in_buf[i]);
    ctx->in_buf_count = 0;
    if (ctx->context) {
        aclrtDestroyContext(ctx->context);
        ctx->context = NULL;
    }
    ascend_enc_global_uninit(ctx->device_id);
    return 0;
}

#define OFFSET(x) offsetof(AscendEncContext, x)
#define VE (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption ascend_encode_options[] = {
    { "device_id", "Ascend NPU device id (-1 = auto-pick first available)", OFFSET(device_id),
      AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 31, VE },
    { "rc_mode", "rate control mode", OFFSET(rc_mode),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 3, VE, .unit = "rc" },
    { "cbr",   "constant bitrate",  0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, VE, .unit = "rc" },
    { "vbr",   "variable bitrate",  0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, VE, .unit = "rc" },
    { "avbr",  "average bitrate",   0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, VE, .unit = "rc" },
    { "fixqp", "fixed QP",          0, AV_OPT_TYPE_CONST, { .i64 = 3 }, 0, 0, VE, .unit = "rc" },
    { "gop",   "GOP size", OFFSET(gop),
      AV_OPT_TYPE_INT, { .i64 = 250 }, 1, 65536, VE },
    { "qp",    "fixed I-frame QP (fixqp mode)", OFFSET(qp),
      AV_OPT_TYPE_INT, { .i64 = 30 }, 0, 51, VE },
    { "zero_copy", "feed ASCEND hw frames to VENC without staging copy "
                   "(requires -output_mode hw decode upstream)",
      OFFSET(zero_copy), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VE },
    { NULL }
};

static const AVClass ascend_encode_class = {
    .class_name = "ascend encoder",
    .item_name  = av_default_item_name,
    .option     = ascend_encode_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

#define DECLARE_ASCEND_ENC(x, X, long_name)                                    \
const FFCodec ff_##x##_ascend_encoder = {                                      \
    .p.name          = #x "_ascend",                                           \
    CODEC_LONG_NAME(long_name " via Ascend NPU (CANN DVPP)"),                   \
    .p.type          = AVMEDIA_TYPE_VIDEO,                                     \
    .p.id            = AV_CODEC_ID_##X,                                        \
    .priv_data_size  = sizeof(AscendEncContext),                               \
    .init            = ascend_encode_init,                                     \
    FF_CODEC_ENCODE_CB(ascend_encode_cb),                                      \
    .close           = ascend_encode_close,                                    \
    .p.capabilities  = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_DR1,               \
    .caps_internal   = FF_CODEC_CAP_INIT_CLEANUP,                               \
    .p.priv_class    = &ascend_encode_class,                                   \
    .p.wrapper_name  = "ascend",                                               \
    CODEC_PIXFMTS(AV_PIX_FMT_NV12,                                             \
                  AV_PIX_FMT_ASCEND),                                          \
};

DECLARE_ASCEND_ENC(h264, H264, "H.264")
DECLARE_ASCEND_ENC(hevc, HEVC, "H.265")
