// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>

#include "common/assert.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"

namespace Metal {

namespace {

/// Idle buffers a level keeps before TickFrame starts releasing them.
constexpr size_t KEPT_IDLE_BUFFERS_PER_LEVEL = 4;

u32 Log2Ceil(size_t size) {
    return static_cast<u32>(std::bit_width(std::max<size_t>(size, 1) - 1));
}

} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    if (StagingBuffer* const reused = TryReuse(size, usage, deferred)) {
        return reused->Ref();
    }
    return Create(size, usage, deferred).Ref();
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    auto& level = LevelsFor(ref.usage)[ref.log2_level];
    const auto it = std::ranges::find_if(
        level, [&ref](const StagingBuffer& entry) { return entry.index == ref.index; });
    ASSERT(it != level.end() && it->deferred);
    it->deferred = false;
    it->tick = scheduler.CurrentTick();
}

void StagingBufferPool::TickFrame() {
    // Visit one level per frame and drop idle buffers above the kept amount.
    release_level = (release_level + 1) % NUM_LEVELS;
    for (Levels* const levels : {&upload_levels, &download_levels}) {
        auto& level = (*levels)[release_level];
        size_t idle = 0;
        std::erase_if(level, [&](const StagingBuffer& entry) {
            if (entry.deferred || !scheduler.IsFree(entry.tick)) {
                return false;
            }
            return ++idle > KEPT_IDLE_BUFFERS_PER_LEVEL;
        });
    }
}

u64 StagingBufferPool::GetMemoryUsage() const {
    u64 total = 0;
    for (const Levels* const levels : {&upload_levels, &download_levels}) {
        for (const auto& level : *levels) {
            for (const StagingBuffer& entry : level) {
                total += entry.mapped_span.size();
            }
        }
    }
    return total;
}

StagingBufferPool::Levels& StagingBufferPool::LevelsFor(MemoryUsage usage) {
    return usage == MemoryUsage::Upload ? upload_levels : download_levels;
}

StagingBufferPool::StagingBuffer* StagingBufferPool::TryReuse(size_t size, MemoryUsage usage,
                                                              bool deferred) {
    auto& level = LevelsFor(usage)[Log2Ceil(size)];
    const auto it = std::ranges::find_if(level, [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    });
    if (it == level.end()) {
        return nullptr;
    }
    it->tick = deferred ? std::numeric_limits<u64>::max() : scheduler.CurrentTick();
    it->deferred = deferred;
    return &*it;
}

StagingBufferPool::StagingBuffer& StagingBufferPool::Create(size_t size, MemoryUsage usage,
                                                            bool deferred) {
    const u32 log2 = Log2Ceil(size);
    const size_t capacity = size_t{1} << log2;
    if (capacity > device.GetMaxBufferLength()) {
        throw std::bad_alloc();
    }
    // Uploads are only written by the CPU, so write-combined CPU caching is faster for them.
    const MTLResourceOptions options =
        MTLResourceStorageModeShared | (usage == MemoryUsage::Upload
                                            ? MTLResourceCPUCacheModeWriteCombined
                                            : MTLResourceCPUCacheModeDefaultCache);
    id<MTLBuffer> buffer = [device.GetDevice() newBufferWithLength:capacity options:options];
    if (buffer == nil) {
        throw std::bad_alloc();
    }
    buffer.label = usage == MemoryUsage::Upload ? @"staging upload" : @"staging download";

    auto& level = LevelsFor(usage)[log2];
    return level.emplace_back(StagingBuffer{
        .buffer = buffer,
        .mapped_span = std::span(static_cast<u8*>(buffer.contents), capacity),
        .usage = usage,
        .log2_level = log2,
        .index = buffer_index++,
        .tick = deferred ? std::numeric_limits<u64>::max() : scheduler.CurrentTick(),
        .deferred = deferred,
    });
}

} // namespace Metal
