/*
 * Ascend NPU (CANN DVPP) hardware device context.
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

#include "config.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>

#include "buffer.h"
#include "common.h"
#include "hwcontext.h"
#include "hwcontext_ascend.h"
#include "hwcontext_internal.h"
#include "imgutils.h"
#include "log.h"
#include "mem.h"
#include "pixdesc.h"
#include "pixfmt.h"

/* aclInit is a process singleton shared with libavcodec. */
static pthread_mutex_t ascend_ref_lock = PTHREAD_MUTEX_INITIALIZER;
static int ascend_ref_count;

static void ascend_global_ref(void)
{
    pthread_mutex_lock(&ascend_ref_lock);
    if (ascend_ref_count++ == 0)
        /* Fails benignly when ACL is already initialized (e.g. torch_npu);
         * subsequent ACL calls surface real errors. */
        aclInit(NULL);
    pthread_mutex_unlock(&ascend_ref_lock);
}

static void ascend_global_unref(void)
{
    pthread_mutex_lock(&ascend_ref_lock);
    if (--ascend_ref_count == 0)
        aclFinalize();
    pthread_mutex_unlock(&ascend_ref_lock);
}

static int ascend_bytes_per_sample(enum AVPixelFormat sw_fmt)
{
    return sw_fmt == AV_PIX_FMT_P010LE ? 2 : 1;
}

/* Stride rules: VENC/VPC want the width stride in bytes 16-aligned
 * (32-min) and the height 2-aligned; VDEC output additionally wants a
 * 16-aligned height, so use the max of both (16) there — handled by the
 * decoder, this pool only guarantees the VPC/VENC rules. */
static int ascend_width_stride(enum AVPixelFormat sw_fmt, int width)
{
    const int bps = ascend_bytes_per_sample(sw_fmt);

    return FFALIGN(width * bps, bps > 1 ? 32 : 16);
}

static int ascend_height_stride(enum AVPixelFormat sw_fmt, int height)
{
    return FFALIGN(height, 2);
}

static size_t ascend_frame_size(enum AVPixelFormat sw_fmt, int width, int height)
{
    return (size_t)ascend_width_stride(sw_fmt, width) *
           ascend_height_stride(sw_fmt, height) * 3 / 2;
}

/* Per-device state; device_init runs once per AVHWDeviceContext. */
typedef struct AscendDeviceContext {
    AVAscendDeviceContext p;

    int context_created;
    int dvpp_inited;
    int global_inited;
} AscendDeviceContext;

static void ascend_device_uninit(AVHWDeviceContext *hwdev)
{
    AscendDeviceContext *dev = hwdev->hwctx;

    if (dev->dvpp_inited) {
        hi_mpi_sys_exit();
        dev->dvpp_inited = 0;
    }
    if (dev->context_created) {
        aclrtDestroyContext(dev->p.context);
        dev->context_created = 0;
    }
    aclrtResetDevice(dev->p.device_id >= 0 ? dev->p.device_id : 0);
    if (dev->global_inited) {
        ascend_global_unref();
        dev->global_inited = 0;
    }
}

static int ascend_device_init(AVHWDeviceContext *hwdev)
{
    AscendDeviceContext *dev = hwdev->hwctx;
    int ret;

    if (dev->global_inited)
        return 0;
    if (dev->p.device_id < 0)
        dev->p.device_id = 0;

    ascend_global_ref();
    dev->global_inited = 1;

    ret = aclrtSetDevice(dev->p.device_id);
    if (ret != 0) {
        av_log(hwdev, AV_LOG_ERROR, "ascend: aclrtSetDevice(%d) failed: %d\n",
               dev->p.device_id, ret);
        goto fail;
    }
    ret = aclrtCreateContext(&dev->p.context, dev->p.device_id);
    if (ret != 0) {
        av_log(hwdev, AV_LOG_ERROR, "ascend: aclrtCreateContext failed: %d\n", ret);
        aclrtResetDevice(dev->p.device_id);
        goto fail;
    }
    dev->context_created = 1;
    aclrtSetCurrentContext(dev->p.context);

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(hwdev, AV_LOG_ERROR, "ascend: hi_mpi_sys_init failed: 0x%x\n", ret);
        goto fail;
    }
    dev->dvpp_inited = 1;

    av_log(hwdev, AV_LOG_VERBOSE, "ascend: device %d initialized\n",
           dev->p.device_id);
    return 0;

fail:
    ascend_device_uninit(hwdev);
    return AVERROR_EXTERNAL;
}

static int ascend_device_create(AVHWDeviceContext *hwdev, const char *device,
                                AVDictionary *opts, int flags)
{
    AscendDeviceContext *dev = hwdev->hwctx;

    dev->p.device_id = 0;
    if (device && device[0]) {
        char *end;
        long id = strtol(device, &end, 10);
        if (*end != '\0' || id < 0 || id > 31) {
            av_log(hwdev, AV_LOG_ERROR, "ascend: invalid device \"%s\"\n", device);
            return AVERROR(EINVAL);
        }
        dev->p.device_id = id;
    }
    return 0;
}

static int ascend_frames_get_constraints(AVHWDeviceContext *hwdev,
                                         const void *hwconfig,
                                         AVHWFramesConstraints *constraints)
{
    constraints->valid_sw_formats = av_malloc_array(3, sizeof(enum AVPixelFormat));
    if (!constraints->valid_sw_formats)
        return AVERROR(ENOMEM);
    constraints->valid_sw_formats[0] = AV_PIX_FMT_NV12;
    constraints->valid_sw_formats[1] = AV_PIX_FMT_P010LE;
    constraints->valid_sw_formats[2] = AV_PIX_FMT_NONE;

    constraints->valid_hw_formats = av_malloc_array(2, sizeof(enum AVPixelFormat));
    if (!constraints->valid_hw_formats) {
        av_freep(&constraints->valid_sw_formats);
        return AVERROR(ENOMEM);
    }
    constraints->valid_hw_formats[0] = AV_PIX_FMT_ASCEND;
    constraints->valid_hw_formats[1] = AV_PIX_FMT_NONE;

    constraints->min_width  = 128;
    constraints->min_height = 128;
    constraints->max_width  = 4096;
    constraints->max_height = 4096;
    return 0;
}

/* A pool buffer carries the DVPP device pointer in a small host wrapper
 * so the free callback can hand it back to the DVPP allocator. */
typedef struct AscendSurface {
    void *dev_ptr;
} AscendSurface;


static void ascend_surface_free(void *opaque, uint8_t *data)
{
    AscendSurface *surf = (AscendSurface *)data;

    if (surf->dev_ptr)
        hi_mpi_dvpp_free(surf->dev_ptr);
    av_free(surf);
}

static AVBufferRef *ascend_pool_alloc(void *opaque, size_t size)
{
    AVHWFramesContext *hwfc = opaque;
    AscendDeviceContext *dev = hwfc->device_ctx->hwctx;
    AscendSurface *surf;
    AVBufferRef *ref;

    /* Pool growth can happen on foreign threads; bind the device context
     * for the DVPP allocator. */
    aclrtSetCurrentContext(dev->p.context);

    surf = av_mallocz(sizeof(*surf));
    if (!surf)
        return NULL;

    if (hi_mpi_dvpp_malloc(dev->p.device_id, &surf->dev_ptr, size) != 0) {
        av_log(hwfc, AV_LOG_ERROR, "ascend: DVPP buffer alloc (%zu) failed\n", size);
        av_free(surf);
        return NULL;
    }

    ref = av_buffer_create((uint8_t *)surf, sizeof(*surf),
                           ascend_surface_free, NULL, 0);
    if (!ref) {
        hi_mpi_dvpp_free(surf->dev_ptr);
        av_free(surf);
        return NULL;
    }
    return ref;
}

static int ascend_frames_init(AVHWFramesContext *hwfc)
{
    FFHWFramesContext *ctxi = ffhwframesctx(hwfc);
    AscendDeviceContext *dev = hwfc->device_ctx->hwctx;
    AVAscendFramesContext *fctx = hwfc->hwctx;

    switch (hwfc->sw_format) {
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_P010LE:
        break;
    default:
        av_log(hwfc, AV_LOG_ERROR, "ascend: unsupported sw format %s\n",
               av_get_pix_fmt_name(hwfc->sw_format));
        return AVERROR(ENOSYS);
    }

    if (fctx)
        fctx->device_id = dev->p.device_id;

    ctxi->pool_internal = av_buffer_pool_init2(
        ascend_frame_size(hwfc->sw_format, hwfc->width, hwfc->height),
        hwfc, ascend_pool_alloc, NULL);
    if (!ctxi->pool_internal)
        return AVERROR(ENOMEM);

    return 0;
}

static void ascend_frames_uninit(AVHWFramesContext *hwfc)
{
    FFHWFramesContext *ctxi = ffhwframesctx(hwfc);

    av_buffer_pool_uninit(&ctxi->pool_internal);
}

static int ascend_get_buffer(AVHWFramesContext *hwfc, AVFrame *frame)
{
    AscendSurface *surf;
    int wstride = ascend_width_stride(hwfc->sw_format, hwfc->width);
    int hstride = ascend_height_stride(hwfc->sw_format, hwfc->height);

    frame->buf[0] = av_buffer_pool_get(hwfc->pool);
    if (!frame->buf[0])
        return AVERROR(ENOMEM);

    surf          = (AscendSurface *)frame->buf[0]->data;
    frame->data[0] = surf->dev_ptr;
    frame->data[1] = (uint8_t *)surf->dev_ptr + (size_t)wstride * hstride;
    frame->data[3] = (uint8_t *)surf;
    frame->format  = AV_PIX_FMT_ASCEND;
    frame->width   = hwfc->width;
    frame->height  = hwfc->height;
    frame->linesize[0] = wstride;
    frame->linesize[1] = wstride;

    return 0;
}

static int ascend_transfer_get_formats(AVHWFramesContext *hwfc,
                                       enum AVHWFrameTransferDirection dir,
                                       enum AVPixelFormat **formats)
{
    enum AVPixelFormat *fmts;

    fmts = av_malloc_array(2, sizeof(*fmts));
    if (!fmts)
        return AVERROR(ENOMEM);
    fmts[0] = hwfc->sw_format;
    fmts[1] = AV_PIX_FMT_NONE;
    *formats = fmts;
    return 0;
}

static int ascend_transfer_data_from(AVHWFramesContext *hwfc,
                                     AVFrame *dst, const AVFrame *src)
{
    const int bps = ascend_bytes_per_sample(hwfc->sw_format);
    const int wstride = ascend_width_stride(hwfc->sw_format, src->width);
    const int hstride = ascend_height_stride(hwfc->sw_format, src->height);
    const uint8_t *dev_base = (const uint8_t *)src->data[0];
    const int rows[2] = { src->height, (src->height + 1) / 2 };
    const int bw = src->width * bps;
    int ret = 0;

    if (!dev_base)
        return AVERROR(EINVAL);

    /* aclrtMemcpy requires the device context of the calling thread. */
    {
        AscendDeviceContext *dev = hwfc->device_ctx->hwctx;
        int arc = aclrtSetCurrentContext(dev->p.context);
        if (arc != 0)
            av_log(hwfc, AV_LOG_WARNING, "ascend: SetCurrentContext %d\n", arc);
    }

    /* When the AVFrame linesize equals the device width stride (the
     * common case), each plane is contiguous on both sides: copy Y and
     * UV as two bulk memcpys using the actual host plane sizes. */
    if (dst->linesize[0] == wstride && dst->linesize[1] == wstride) {
        const size_t y_bytes = (size_t)wstride * hstride;
        const size_t c_bytes = (size_t)wstride * (hstride / 2);

        if (aclrtMemcpy(dst->data[0], y_bytes, dev_base, y_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0)
            return AVERROR_EXTERNAL;
        if (aclrtMemcpy(dst->data[1], c_bytes,
                        dev_base + y_bytes, c_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0)
            return AVERROR_EXTERNAL;
        return 0;
    }

    for (int p = 0; p < 2 && !ret; p++) {
        const uint8_t *src_p = dev_base + (p == 0 ? 0 : (size_t)wstride * hstride);
        uint8_t *dst_p = p == 0 ? dst->data[0] : dst->data[1];
        const int dl = p == 0 ? dst->linesize[0] : dst->linesize[1];

        for (int y = 0; y < rows[p]; y++)
            if (aclrtMemcpy(dst_p + (size_t)y * dl, bw,
                            src_p + (size_t)y * wstride, bw,
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                ret = AVERROR_EXTERNAL;
                break;
            }
    }
    return ret;
}

static int ascend_transfer_data_to(AVHWFramesContext *hwfc,
                                   AVFrame *dst, const AVFrame *src)
{
    const int bps = ascend_bytes_per_sample(hwfc->sw_format);
    const int wstride = ascend_width_stride(hwfc->sw_format, dst->width);
    const int hstride = ascend_height_stride(hwfc->sw_format, dst->height);
    uint8_t *dev_base = (uint8_t *)dst->data[0];
    const int rows[2] = { dst->height, (dst->height + 1) / 2 };
    const int bw = dst->width * bps;
    int ret = 0;

    if (!dev_base)
        return AVERROR(EINVAL);

    {
        AscendDeviceContext *dev = hwfc->device_ctx->hwctx;
        int arc = aclrtSetCurrentContext(dev->p.context);
        if (arc != 0)
            av_log(hwfc, AV_LOG_WARNING, "ascend: SetCurrentContext %d\n", arc);
    }
    for (int p = 0; p < 2 && !ret; p++) {
        uint8_t *dst_p = dev_base + (p == 0 ? 0 : (size_t)wstride * hstride);
        const uint8_t *src_p = p == 0 ? src->data[0] : src->data[1];
        const int sl = p == 0 ? src->linesize[0] : src->linesize[1];

        for (int y = 0; y < rows[p]; y++)
            if (aclrtMemcpy(dst_p + (size_t)y * wstride, bw,
                            src_p + (size_t)y * sl, bw,
                            ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
                ret = AVERROR_EXTERNAL;
                break;
            }
    }
    return ret;
}

const HWContextType ff_hwcontext_type_ascend = {
    .type                   = AV_HWDEVICE_TYPE_ASCEND,
    .name                   = "ascend",

    .device_hwctx_size      = sizeof(AscendDeviceContext),

    .device_create          = &ascend_device_create,
    .device_init            = &ascend_device_init,
    .device_uninit          = &ascend_device_uninit,
    .frames_get_constraints = &ascend_frames_get_constraints,
    .frames_init            = &ascend_frames_init,
    .frames_uninit          = &ascend_frames_uninit,
    .frames_get_buffer      = &ascend_get_buffer,
    .transfer_get_formats   = &ascend_transfer_get_formats,
    .transfer_data_from     = &ascend_transfer_data_from,
    .transfer_data_to       = &ascend_transfer_data_to,

    .pix_fmts = (const enum AVPixelFormat[]) {
        AV_PIX_FMT_ASCEND,
        AV_PIX_FMT_NONE
    },
};
