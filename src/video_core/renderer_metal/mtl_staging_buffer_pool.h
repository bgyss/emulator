// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_staging_buffer_pool.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Metal {

class Device;
class Scheduler;

enum class MemoryUsage {
    /// Written by the CPU, read by the GPU.
    Upload,
    /// Written by the GPU, read by the CPU.
    Download,
};

struct StagingBufferRef {
    id<MTLBuffer> buffer;
    size_t offset{};
    std::span<u8> mapped_span;
    MemoryUsage usage{};
    u32 log2_level{};
    u64 index{};
};

/**
 * Hands out CPU-visible MTLBuffers for transfers, like the Vulkan renderer's StagingBufferPool.
 *
 * Buffers come in power-of-two sizes. A buffer is reused once the GPU has finished the tick it
 * was last requested on, unless it was requested as deferred: then it stays reserved until
 * FreeDeferred, for transfers whose result the CPU reads back later.
 */
class StagingBufferPool {
public:
    explicit StagingBufferPool(const Device& device, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    [[nodiscard]] StagingBufferRef Request(size_t size, MemoryUsage usage, bool deferred = false);

    /// Releases a buffer requested as deferred; it becomes reusable after the current tick.
    void FreeDeferred(StagingBufferRef& ref);

    /// Releases a few idle buffers so the pool shrinks after a spike. Call once per frame.
    void TickFrame();

    /// Total bytes held by the pool.
    [[nodiscard]] u64 GetMemoryUsage() const;

private:
    static constexpr size_t NUM_LEVELS = sizeof(size_t) * 8;

    struct StagingBuffer {
        id<MTLBuffer> buffer;
        std::span<u8> mapped_span;
        MemoryUsage usage{};
        u32 log2_level{};
        u64 index{};
        u64 tick{};
        bool deferred{};

        [[nodiscard]] StagingBufferRef Ref() const {
            return {
                .buffer = buffer,
                .offset = 0,
                .mapped_span = mapped_span,
                .usage = usage,
                .log2_level = log2_level,
                .index = index,
            };
        }
    };

    using Levels = std::array<std::vector<StagingBuffer>, NUM_LEVELS>;

    [[nodiscard]] Levels& LevelsFor(MemoryUsage usage);

    StagingBuffer* TryReuse(size_t size, MemoryUsage usage, bool deferred);
    StagingBuffer& Create(size_t size, MemoryUsage usage, bool deferred);

    const Device& device;
    Scheduler& scheduler;
    Levels upload_levels;
    Levels download_levels;
    u64 buffer_index{};
    size_t release_level{};
};

} // namespace Metal
