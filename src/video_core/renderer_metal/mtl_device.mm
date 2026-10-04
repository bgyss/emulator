// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <TargetConditionals.h>

#include <stdexcept>

#include "common/logging.h"
#include "video_core/renderer_metal/mtl_device.h"

namespace Metal {

Device::Device() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("No Metal device available");
    }
    queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("Failed to create a Metal command queue");
    }
    queue.label = @"citron";

    name = device.name.UTF8String;
    has_unified_memory = device.hasUnifiedMemory;
    is_apple_silicon = [device supportsFamily:MTLGPUFamilyApple7];
    supports_astc = [device supportsFamily:MTLGPUFamilyApple2];
#if TARGET_OS_OSX
    supports_bc = device.supportsBCTextureCompression;
#else
    supports_bc = false;
#endif
    recommended_working_set_size = device.recommendedMaxWorkingSetSize;
    max_buffer_length = device.maxBufferLength;

    LOG_INFO(Render_Metal,
             "Metal device: {} (unified memory: {}, Apple7+: {}, BC: {}, ASTC: {}, working set: "
             "{} MiB, max buffer: {} MiB)",
             name, has_unified_memory, is_apple_silicon, supports_bc, supports_astc,
             recommended_working_set_size >> 20, max_buffer_length >> 20);
}

Device::~Device() = default;

} // namespace Metal
