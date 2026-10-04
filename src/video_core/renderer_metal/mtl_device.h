// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_device.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <string>

#include "common/common_types.h"

namespace Metal {

/// Owns the Metal device and the command queue all Metal work is submitted to, and reports the
/// capabilities the renderer needs to choose code paths.
class Device {
public:
    /// @throws std::runtime_error when no Metal device or command queue is available.
    Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] id<MTLDevice> GetDevice() const {
        return device;
    }

    [[nodiscard]] id<MTLCommandQueue> GetQueue() const {
        return queue;
    }

    [[nodiscard]] const std::string& GetName() const {
        return name;
    }

    /// CPU and GPU share memory (Apple silicon), so shared-storage resources are as fast as
    /// private ones for most uses.
    [[nodiscard]] bool HasUnifiedMemory() const {
        return has_unified_memory;
    }

    /// Apple7 (M1/A14) or newer.
    [[nodiscard]] bool IsAppleSilicon() const {
        return is_apple_silicon;
    }

    [[nodiscard]] bool SupportsBCTextureCompression() const {
        return supports_bc;
    }

    [[nodiscard]] bool SupportsASTC() const {
        return supports_astc;
    }

    /// Bytes of GPU-accessible memory the device can use without hurting performance.
    [[nodiscard]] u64 GetRecommendedWorkingSetSize() const {
        return recommended_working_set_size;
    }

    [[nodiscard]] u64 GetMaxBufferLength() const {
        return max_buffer_length;
    }

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::string name;
    bool has_unified_memory{};
    bool is_apple_silicon{};
    bool supports_bc{};
    bool supports_astc{};
    u64 recommended_working_set_size{};
    u64 max_buffer_length{};
};

} // namespace Metal
