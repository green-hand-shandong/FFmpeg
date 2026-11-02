/*
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

#ifndef AVUTIL_HWCONTEXT_ASCEND_H
#define AVUTIL_HWCONTEXT_ASCEND_H

#include <acl/acl.h>

#include "pixfmt.h"

/**
 * @file
 * An AVHWDeviceContext for Ascend NPU DVPP (CANN) device memory.
 */

/**
 * Ascend-specific device context, allocated as AVHWDeviceContext.hwctx.
 * The device id selects the NPU chip (as in npu-smi); contexts and the
 * DVPP subsystem are initialized per process on first use.
 */
typedef struct AVAscendDeviceContext {
    /**
     * Ascend NPU device id (-1 = unspecified; defaults to 0 on use).
     */
    int device_id;

    /**
     * ACL context created by the device (NULL before av_hwdevice_ctx_init).
     * Foreign threads must aclrtSetCurrentContext(this) before ACL/DVPP
     * calls touching buffers of this device.
     */
    aclrtContext context;
} AVAscendDeviceContext;

/**
 * Ascend-specific frame pool context, allocated as AVHWFramesContext.hwctx.
 * Surfaces are plain DVPP device buffers of frames->width x frames->height
 * in the pool's sw_format; stride metadata lives in AVHWFramesContext
 * (line sizes are derived from the width with CANN alignment rules).
 */
typedef struct AVAscendFramesContext {
    /**
     * Device the pool buffers were allocated on; inherited from the
     * device context at pool init time.
     */
    int device_id;
} AVAscendFramesContext;

#endif /* AVUTIL_HWCONTEXT_ASCEND_H */
