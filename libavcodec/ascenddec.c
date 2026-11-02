/*
 * Ascend NPU (Atlas 300I-Duo / Ascend 310P) hardware video decoder
 * via the CANN DVPP media APIs (hi_mpi_vdec).
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

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/error.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"

#define ASCEND_MAX_CHN        32
#define ASCEND_IN_FLIGHT_SIZE 64
#define ASCEND_OUT_POOL_SIZE  16

/* Process-wide state: aclInit() is a process singleton and VDEC channel ids
 * must be unique within a process. */
static pthread_mutex_t ascend_global_lock = PTHREAD_MUTEX_INITIALIZER;
static int ascend_global_ref;
static uint32_t ascend_chn_used[(ASCEND_MAX_CHN + 31) / 32];

typedef struct AscendDecContext {
    AVClass *class;

    int device_id;

    int global_inited;
    int dvpp_inited;
    int chn_created;
    hi_vdec_chn chn;
    aclrtContext context;
    AVCodecContext *avctx;

    int coded_width;
    int coded_height;
    int bit_depth;       /* source bit depth: 8 or 10 */
    int bytes_per_sample; /* 1 (8-bit NV12) or 2 (10-bit P010) */
    int bit_depth_out;   /* bytes per sample: 1 (NV12) or 2 (P010) */
    int out_width_stride;
    int out_height_stride;
    size_t out_buffer_size;
    size_t stream_buf_size;

    /* In-flight DVPP input buffers. The decoder keeps referencing a stream
     * buffer until its access unit has been decoded, so a buffer may only be
     * freed once hi_vdec_chn_status.dec_stream_frames has passed its send
     * index (see ascend_free_consumed_inputs). */
    struct {
        void *buf;
        uint64_t send_index;
    } in_flight[ASCEND_IN_FLIGHT_SIZE];
    int in_flight_count;
    uint64_t sends;
    void *eos_buf;

    /* Pool of DVPP output buffers handed to the decoder, one per send.
     * Ownership comes back with each decoded frame. */
    void *out_pool[ASCEND_OUT_POOL_SIZE];
    int out_pool_count;

    int eos_sent;
    int drain_fail_cnt;
    AVPacket *pkt;
} AscendDecContext;

static int ascend_global_init(void)
{
    pthread_mutex_lock(&ascend_global_lock);
    if (ascend_global_ref++ == 0) {
        /* Fails benignly when the hosting process (e.g. torch_npu) already
         * initialized ACL. */
        int ret = aclInit(NULL);
        if (ret != 0)
            av_log(NULL, AV_LOG_DEBUG,
                   "ascenddec: aclInit returned %d (already initialized?)\n", ret);
    }
    pthread_mutex_unlock(&ascend_global_lock);
    return 0;
}

static void ascend_global_uninit(void)
{
    pthread_mutex_lock(&ascend_global_lock);
    if (--ascend_global_ref == 0)
        aclFinalize();
    pthread_mutex_unlock(&ascend_global_lock);
}

static int ascend_chn_alloc(hi_vdec_chn *chn)
{
    int ret = AVERROR(ENOSYS);

    pthread_mutex_lock(&ascend_global_lock);
    for (int i = 0; i < ASCEND_MAX_CHN; i++) {
        if (!(ascend_chn_used[i / 32] & (1u << (i % 32)))) {
            ascend_chn_used[i / 32] |= 1u << (i % 32);
            *chn = i;
            ret = 0;
            break;
        }
    }
    pthread_mutex_unlock(&ascend_global_lock);
    return ret;
}

static void ascend_chn_free(hi_vdec_chn chn)
{
    pthread_mutex_lock(&ascend_global_lock);
    ascend_chn_used[chn / 32] &= ~(1u << (chn % 32));
    pthread_mutex_unlock(&ascend_global_lock);
}

static void ascend_release_device(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;

    for (int i = 0; i < ctx->in_flight_count; i++)
        hi_mpi_dvpp_free(ctx->in_flight[i].buf);
    ctx->in_flight_count = 0;
    if (ctx->eos_buf) {
        hi_mpi_dvpp_free(ctx->eos_buf);
        ctx->eos_buf = NULL;
    }
    for (int i = 0; i < ASCEND_OUT_POOL_SIZE; i++)
        if (ctx->out_pool[i]) {
            hi_mpi_dvpp_free(ctx->out_pool[i]);
            ctx->out_pool[i] = NULL;
        }
    if (ctx->context) {
        aclrtDestroyContext(ctx->context);
        ctx->context = NULL;
    }
    aclrtResetDevice(ctx->device_id);
}

static av_cold int ascend_channel_create(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;
    hi_payload_type type = avctx->codec_id == AV_CODEC_ID_HEVC ?
                           HI_PT_H265 : HI_PT_H264;
    hi_vdec_chn_attr attr = { 0 };
    hi_vdec_chn_param chn_param = { 0 };
    hi_pic_buf_attr buf_attr = { 0 };
    int ret;

    ctx->coded_width       = avctx->coded_width;
    ctx->coded_height      = avctx->coded_height;
    /* 10-bit streams (H.264 High 10 / HEVC Main 10) decode to 2-byte-per-
     * sample semi-planar output (P010 layout); strides are in bytes. */
    ctx->bytes_per_sample  = ctx->bit_depth > 8 ? 2 : 1;
    ctx->out_width_stride  = FFALIGN(ctx->coded_width * ctx->bytes_per_sample,
                                     ctx->bytes_per_sample > 1 ? 32 : 16);
    ctx->out_height_stride = FFALIGN(ctx->coded_height, 16);
    ctx->out_buffer_size   = (size_t)ctx->out_width_stride *
                             ctx->out_height_stride * 3 / 2;
    ctx->stream_buf_size   = FFMAX(1024 * 1024, ctx->out_buffer_size);

    ret = aclrtSetDevice(ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: aclrtSetDevice(%d) failed: %d\n",
               ctx->device_id, ret);
        return AVERROR_EXTERNAL;
    }
    ret = aclrtCreateContext(&ctx->context, ctx->device_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: aclrtCreateContext failed: %d\n", ret);
        aclrtResetDevice(ctx->device_id);
        return AVERROR_EXTERNAL;
    }
    aclrtSetCurrentContext(ctx->context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: hi_mpi_sys_init failed: 0x%x\n", ret);
        ascend_release_device(avctx);
        return AVERROR_EXTERNAL;
    }
    ctx->dvpp_inited = 1;

    ret = ascend_chn_alloc(&ctx->chn);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: no free VDEC channel\n");
        goto fail;
    }

    attr.type            = type;
    attr.mode            = HI_VDEC_SEND_MODE_FRAME;
    attr.pic_width       = ctx->coded_width;
    attr.pic_height      = ctx->coded_height;
    attr.stream_buf_size = ctx->stream_buf_size;
    attr.frame_buf_cnt   = 8 + 2 + 1;
    buf_attr.width         = ctx->coded_width;
    buf_attr.height        = ctx->coded_height;
    buf_attr.bit_width     = ctx->bit_depth > 8 ? HI_DATA_BIT_WIDTH_10 :
                                                  HI_DATA_BIT_WIDTH_8;
    buf_attr.pixel_format  = ctx->bit_depth > 8 ?
                             HI_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT :
                             HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    buf_attr.compress_mode = HI_COMPRESS_MODE_NONE;
    attr.frame_buf_size    = hi_vdec_get_pic_buf_size(type, &buf_attr);
    attr.video_attr.ref_frame_num   = 8;
    /* tmv_buf_size must be non-zero for all stream types, otherwise
     * hi_mpi_vdec_create_chn fails with 0xa0058003. */
    attr.video_attr.temporal_mvp_en = HI_TRUE;
    attr.video_attr.tmv_buf_size =
        hi_vdec_get_tmv_buf_size(type, ctx->coded_width, ctx->coded_height);

    ret = hi_mpi_vdec_create_chn(ctx->chn, &attr);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR,
               "ascenddec: hi_mpi_vdec_create_chn(%d) failed: 0x%x "
               "(%dx%d stream_buf=%u frame_buf=%u cnt=%u ref=%u)\n",
               ctx->chn, ret, attr.pic_width, attr.pic_height,
               attr.stream_buf_size, attr.frame_buf_size, attr.frame_buf_cnt,
               attr.video_attr.ref_frame_num);
        ascend_chn_free(ctx->chn);
        goto fail;
    }

    ret = hi_mpi_vdec_get_chn_param(ctx->chn, &chn_param);
    if (ret == 0) {
        chn_param.video_param.out_order     = HI_VIDEO_OUT_ORDER_DISPLAY;
        chn_param.video_param.video_format  = HI_VIDEO_FORMAT_LINEAR;
        chn_param.video_param.compress_mode = HI_COMPRESS_MODE_NONE;
        ret = hi_mpi_vdec_set_chn_param(ctx->chn, &chn_param);
    }
    if (ret != 0)
        av_log(avctx, AV_LOG_WARNING, "ascenddec: hi_mpi_vdec_set_chn_param failed: 0x%x\n", ret);

    ret = hi_mpi_vdec_start_recv_stream(ctx->chn);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: hi_mpi_vdec_start_recv_stream failed: 0x%x\n", ret);
        hi_mpi_vdec_destroy_chn(ctx->chn);
        ascend_chn_free(ctx->chn);
        goto fail;
    }

    for (int i = 0; i < ASCEND_OUT_POOL_SIZE; i++) {
        if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->out_pool[i],
                               ctx->out_buffer_size) != 0) {
            av_log(avctx, AV_LOG_ERROR, "ascenddec: hi_mpi_dvpp_malloc failed\n");
            goto fail;
        }
        ctx->out_pool_count++;
    }
    if (hi_mpi_dvpp_malloc(ctx->device_id, &ctx->eos_buf, 4096) != 0) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: hi_mpi_dvpp_malloc failed\n");
        goto fail;
    }

    ctx->chn_created = 1;
    av_log(avctx, AV_LOG_VERBOSE, "ascenddec: channel %d created, %dx%d on device %d\n",
           ctx->chn, ctx->coded_width, ctx->coded_height, ctx->device_id);
    return 0;

fail:
    hi_mpi_vdec_stop_recv_stream(ctx->chn);
    hi_mpi_vdec_destroy_chn(ctx->chn);
    ascend_chn_free(ctx->chn);
    hi_mpi_sys_exit();
    ctx->dvpp_inited = 0;
    ascend_release_device(avctx);
    return AVERROR_EXTERNAL;
}

static av_cold int ascend_decode_init(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;

    ctx->avctx = avctx;

    if (avctx->pix_fmt == AV_PIX_FMT_NONE)
        avctx->pix_fmt = ctx->bit_depth > 8 ? AV_PIX_FMT_P010LE :
                                              AV_PIX_FMT_NV12;

    ctx->pkt = av_packet_alloc();
    if (!ctx->pkt)
        return AVERROR(ENOMEM);

    ascend_global_init();
    ctx->global_inited = 1;
    return 0;
}

static void ascend_return_out_buffer(AscendDecContext *ctx, void *buf)
{
    if (buf && ctx->out_pool_count < ASCEND_OUT_POOL_SIZE)
        ctx->out_pool[ctx->out_pool_count++] = buf;
}

/* Free input buffers whose access units the decoder has finished with. */
static void ascend_free_consumed_inputs(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;
    hi_vdec_chn_status st = { 0 };

    if (ctx->in_flight_count == 0)
        return;
    if (hi_mpi_vdec_query_status(ctx->chn, &st) != 0)
        return;
    while (ctx->in_flight_count > 0 &&
           (uint64_t)st.dec_stream_frames > ctx->in_flight[0].send_index) {
        hi_mpi_dvpp_free(ctx->in_flight[0].buf);
        ctx->in_flight_count--;
        memmove(&ctx->in_flight[0], &ctx->in_flight[1],
                sizeof(ctx->in_flight[0]) * ctx->in_flight_count);
    }
}

static int ascend_send_stream(AscendDecContext *ctx, uint8_t *buf, int len,
                              uint64_t pts, int eos)
{
    hi_vdec_stream stream = { 0 };
    hi_vdec_pic_info out_pic = { 0 };
    void *out_buf = NULL;
    int ret;

    if (!eos) {
        if (ctx->out_pool_count == 0)
            return AVERROR(EAGAIN);
        out_buf = ctx->out_pool[--ctx->out_pool_count];
    }

    stream.addr          = buf;
    stream.len           = len;
    stream.pts           = pts;
    stream.end_of_frame  = HI_TRUE;
    stream.end_of_stream = eos ? HI_TRUE : HI_FALSE;
    stream.need_display  = eos ? HI_FALSE : HI_TRUE;

    out_pic.width         = eos ? 0 : ctx->coded_width;
    out_pic.height        = eos ? 0 : ctx->coded_height;
    out_pic.width_stride  = eos ? 0 : ctx->out_width_stride;
    out_pic.height_stride = eos ? 0 : ctx->out_height_stride;
    out_pic.pixel_format  = ctx->bit_depth > 8 ?
                            HI_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT :
                            HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    out_pic.vir_addr      = (uint64_t)(uintptr_t)out_buf;
    out_pic.buffer_size   = out_buf ? ctx->out_buffer_size : 0;

    ret = hi_mpi_vdec_send_stream(ctx->chn, &stream, &out_pic, 1000 /* ms */);
    if (ret != 0)
        av_log(ctx->avctx, AV_LOG_ERROR,
               "ascenddec: send_stream failed: 0x%x (len=%u w=%u h=%u "
               "wstride=%u hstride=%u fmt=%d bufsize=%u eos=%d)\n",
               ret, stream.len, out_pic.width, out_pic.height,
               out_pic.width_stride, out_pic.height_stride,
               out_pic.pixel_format, out_pic.buffer_size,
               stream.end_of_stream);

    return ret;
}

/* Fetch one decoded frame into AVFrame.
 * Returns 0 on success, AVERROR(EAGAIN) if nothing is ready. */
static int ascend_get_frame(AVCodecContext *avctx, AVFrame *frame, int timeout_ms)
{
    AscendDecContext *ctx = avctx->priv_data;
    hi_video_frame_info fi = { 0 };
    hi_vdec_supplement_info supp = { 0 };
    hi_vdec_stream stream = { 0 };
    int w;
    uint64_t pts;
    uint8_t *host;
    size_t stride, frame_size;
    int ret;

    ret = hi_mpi_vdec_get_frame(ctx->chn, &fi, &supp, &stream, timeout_ms);
    if (ret != 0)
        return AVERROR(EAGAIN);

    if (fi.v_frame.frame_flag == 2 /* no picture, e.g. second field */) {
        ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
        hi_mpi_vdec_release_frame(ctx->chn, &fi);
        return AVERROR(EAGAIN);
    }
    if (fi.v_frame.frame_flag != 0)
        av_log(avctx, AV_LOG_WARNING, "ascenddec: frame_flag = %u (decode error)\n",
               fi.v_frame.frame_flag);

    w      = fi.v_frame.width;
    pts    = fi.v_frame.pts;
    stride = fi.v_frame.width_stride[0];
    /* 10-bit output is 2 bytes per sample (P010-like high-bit layout). */
    ctx->bit_depth_out = ctx->bit_depth > 8 ? 2 : 1;
    frame_size = stride * (size_t)ctx->out_height_stride * 3 / 2 *
                 ctx->bit_depth_out;

    if (fi.v_frame.width <= 0 || fi.v_frame.height <= 0 || stride < (size_t)fi.v_frame.width ||
        !fi.v_frame.virt_addr[0]) {
        av_log(avctx, AV_LOG_ERROR, "ascenddec: invalid frame from decoder\n");
        ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
        hi_mpi_vdec_release_frame(ctx->chn, &fi);
        return AVERROR_EXTERNAL;
    }

    ret = ff_decode_frame_props(avctx, frame);
    if (ret < 0) {
        ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
        hi_mpi_vdec_release_frame(ctx->chn, &fi);
        return ret;
    }

    frame->buf[0] = av_buffer_alloc(frame_size);
    if (!frame->buf[0]) {
        ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
        hi_mpi_vdec_release_frame(ctx->chn, &fi);
        return AVERROR(ENOMEM);
    }
    host = frame->buf[0]->data;

    {
        /* Single contiguous copy of the strided buffer; the AVFrame keeps
         * the padded linesize, trailing padding is never read. */
        if (aclrtMemcpy(host, frame_size,
                        (void *)(uintptr_t)fi.v_frame.virt_addr[0], frame_size,
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0)
            goto copy_fail;
    }

    ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
    hi_mpi_vdec_release_frame(ctx->chn, &fi);

    frame->data[0]     = host;
    frame->data[1]     = host + stride * (size_t)ctx->out_height_stride *
                         ctx->bit_depth_out;
    frame->linesize[0] = stride * ctx->bit_depth_out;
    frame->linesize[1] = stride * ctx->bit_depth_out;
    frame->width       = fi.v_frame.width;
    frame->height      = fi.v_frame.height;
    frame->format      = ctx->bit_depth > 8 ? AV_PIX_FMT_P010LE :
                                              AV_PIX_FMT_NV12;
    frame->pts         = pts;
    return 0;

copy_fail:
    av_log(avctx, AV_LOG_ERROR, "ascenddec: frame copy failed\n");
    av_buffer_unref(&frame->buf[0]);
    ascend_return_out_buffer(ctx, fi.v_frame.virt_addr[0]);
    hi_mpi_vdec_release_frame(ctx->chn, &fi);
    return AVERROR_EXTERNAL;
}

static int ascend_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    AscendDecContext *ctx = avctx->priv_data;
    int ret;

    if (!ctx->chn_created) {
        if (avctx->coded_width <= 0 || avctx->coded_height <= 0)
            return AVERROR(EAGAIN);
        ret = ascend_channel_create(avctx);
        if (ret < 0)
            return ret;
    }

    for (;;) {
        /* hi_mpi_vdec_get_frame treats a zero timeout as "block forever",
         * so always poll with a small finite timeout. */
        int timeout = ctx->eos_sent ? 20 : (ctx->out_pool_count == 0 ? 100 : 1);

        ret = ascend_get_frame(avctx, frame, timeout);
        if (ret == 0)
            return 0;
        if (ret != AVERROR(EAGAIN))
            return ret;

        if (ctx->eos_sent) {
            hi_vdec_chn_status st = { 0 };

            if (hi_mpi_vdec_query_status(ctx->chn, &st) == 0 &&
                st.left_stream_frames == 0 && st.left_decoded_frames == 0)
                return AVERROR_EOF;
            if (++ctx->drain_fail_cnt > 200) {
                av_log(avctx, AV_LOG_WARNING, "ascenddec: EOS drain timed out\n");
                return AVERROR_EOF;
            }
            continue;
        }

        if (ctx->out_pool_count == 0)
            continue; /* drained above with a timeout until a buffer frees up */

        if (ctx->in_flight_count >= ASCEND_IN_FLIGHT_SIZE)
            continue; /* decoder stalled; drain above until inputs are consumed */

        ret = ff_decode_get_packet(avctx, ctx->pkt);
        if (ret == AVERROR_EOF) {
            ret = ascend_send_stream(ctx, ctx->eos_buf, 0, 0, 1);
            if (ret != 0)
                av_log(avctx, AV_LOG_WARNING, "ascenddec: EOS send failed: 0x%x\n", ret);
            ctx->eos_sent = 1;
            ctx->drain_fail_cnt = 0;
            continue;
        }
        if (ret < 0)
            return ret;

        {
            void *in_buf = NULL;
            const int len = ctx->pkt->size;

            if (hi_mpi_dvpp_malloc(ctx->device_id, &in_buf,
                                   FFMAX(len, 1)) != 0) {
                av_log(avctx, AV_LOG_ERROR, "ascenddec: input dvpp_malloc failed\n");
                av_packet_unref(ctx->pkt);
                return AVERROR(ENOMEM);
            }
            if (len > 0 &&
                aclrtMemcpy(in_buf, len, ctx->pkt->data, len,
                            ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
                av_log(avctx, AV_LOG_ERROR, "ascenddec: input copy failed\n");
                hi_mpi_dvpp_free(in_buf);
                av_packet_unref(ctx->pkt);
                return AVERROR_EXTERNAL;
            }

            ret = ascend_send_stream(ctx, in_buf, len,
                                     ctx->pkt->pts != AV_NOPTS_VALUE ? (uint64_t)ctx->pkt->pts :
                                     ctx->pkt->dts != AV_NOPTS_VALUE ? (uint64_t)ctx->pkt->dts : 0,
                                     0);
            av_packet_unref(ctx->pkt);
            if (ret != 0) {
                av_log(avctx, AV_LOG_ERROR, "ascenddec: send_stream failed: 0x%x\n", ret);
                hi_mpi_dvpp_free(in_buf);
                return AVERROR_EXTERNAL;
            }
            ctx->in_flight[ctx->in_flight_count].buf = in_buf;
            ctx->in_flight[ctx->in_flight_count].send_index = ctx->sends++;
            ctx->in_flight_count++;
        }
        ascend_free_consumed_inputs(avctx);
    }
}

static av_cold int ascend_decode_close(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;

    if (ctx->chn_created) {
        hi_mpi_vdec_stop_recv_stream(ctx->chn);
        hi_mpi_vdec_destroy_chn(ctx->chn);
        ascend_chn_free(ctx->chn);
        ctx->chn_created = 0;
    }
    if (ctx->dvpp_inited) {
        hi_mpi_sys_exit();
        ctx->dvpp_inited = 0;
    }
    ascend_release_device(avctx);
    if (ctx->global_inited) {
        ascend_global_uninit();
        ctx->global_inited = 0;
    }

    av_packet_free(&ctx->pkt);
    return 0;
}

static void ascend_decode_flush(AVCodecContext *avctx)
{
    AscendDecContext *ctx = avctx->priv_data;

    if (!ctx->chn_created)
        return;
    hi_mpi_vdec_stop_recv_stream(ctx->chn);
    hi_mpi_vdec_reset_chn(ctx->chn);
    hi_mpi_vdec_start_recv_stream(ctx->chn);
    for (int i = 0; i < ctx->in_flight_count; i++)
        hi_mpi_dvpp_free(ctx->in_flight[i].buf);
    ctx->in_flight_count = 0;
    ctx->sends           = 0;
    ctx->eos_sent        = 0;
    ctx->drain_fail_cnt  = 0;
    av_packet_unref(ctx->pkt);
}

#define OFFSET(x) offsetof(AscendDecContext, x)
#define VD (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM)
static const AVOption ascend_decode_options[] = {
    { "device_id", "Ascend NPU device id", OFFSET(device_id),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 31, VD },
    { NULL }
};

static const AVClass ascend_decode_class = {
    .class_name = "ascend decoder",
    .item_name  = av_default_item_name,
    .option     = ascend_decode_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

#define DECLARE_ASCEND_DEC(x, X, long_name)                                    \
const FFCodec ff_##x##_ascend_decoder = {                                      \
    .p.name          = #x "_ascend",                                           \
    CODEC_LONG_NAME(long_name " via Ascend NPU (CANN DVPP)"),                   \
    .p.type          = AVMEDIA_TYPE_VIDEO,                                     \
    .p.id            = AV_CODEC_ID_##X,                                        \
    .priv_data_size  = sizeof(AscendDecContext),                               \
    .init            = ascend_decode_init,                                     \
    .close           = ascend_decode_close,                                    \
    .flush           = ascend_decode_flush,                                    \
    FF_CODEC_RECEIVE_FRAME_CB(ascend_receive_frame),                           \
    .p.capabilities  = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_DELAY,             \
    .caps_internal   = FF_CODEC_CAP_SETS_FRAME_PROPS |                         \
                       FF_CODEC_CAP_INIT_CLEANUP,                              \
    .p.priv_class    = &ascend_decode_class,                                   \
    .p.wrapper_name  = "ascend",                                               \
    .bsfs            = #x "_mp4toannexb",                                      \
};

DECLARE_ASCEND_DEC(h264, H264, "H.264")
DECLARE_ASCEND_DEC(hevc, HEVC, "H.265")
