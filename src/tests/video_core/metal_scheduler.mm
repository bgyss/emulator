// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <memory>
#include <numeric>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"

using namespace Metal;

namespace {

std::unique_ptr<Device> MakeDevice() {
    try {
        return std::make_unique<Device>();
    } catch (const std::runtime_error& error) {
        WARN("No Metal device; skipping: " << error.what());
        return nullptr;
    }
}

} // Anonymous namespace

TEST_CASE("Metal scheduler ticks advance and complete", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    REQUIRE(scheduler.CurrentTick() == 1);
    REQUIRE_FALSE(scheduler.IsFree(1));

    const u64 first = scheduler.Flush();
    REQUIRE(first == 1);
    REQUIRE(scheduler.CurrentTick() == 2);
    scheduler.Wait(first);
    REQUIRE(scheduler.IsFree(first));

    // Waiting on the tick still being recorded flushes it.
    scheduler.Wait(scheduler.CurrentTick());
    REQUIRE(scheduler.KnownGpuTick() == 2);
    REQUIRE(scheduler.CurrentTick() == 3);
}

TEST_CASE("Metal scheduler copies between staging buffers", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    StagingBufferPool pool(*device, scheduler);

    constexpr size_t size = 4096;
    const StagingBufferRef upload = pool.Request(size, MemoryUsage::Upload);
    const StagingBufferRef download = pool.Request(size, MemoryUsage::Download);
    REQUIRE(upload.mapped_span.size() >= size);
    std::iota(upload.mapped_span.begin(), upload.mapped_span.begin() + size, u8{0});
    std::fill_n(download.mapped_span.begin(), size, u8{0xCD});

    [scheduler.BlitEncoder() copyFromBuffer:upload.buffer
                               sourceOffset:upload.offset
                                   toBuffer:download.buffer
                          destinationOffset:download.offset
                                       size:size];
    scheduler.Finish();

    for (size_t i = 0; i < size; ++i) {
        REQUIRE(download.mapped_span[i] == static_cast<u8>(i));
    }
}

TEST_CASE("Metal staging buffers are reused only after their tick", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    StagingBufferPool pool(*device, scheduler);

    const StagingBufferRef first = pool.Request(1000, MemoryUsage::Upload);
    REQUIRE(first.mapped_span.size() == 1024);

    // Still in use by the tick being recorded.
    const StagingBufferRef second = pool.Request(1000, MemoryUsage::Upload);
    REQUIRE(second.index != first.index);

    scheduler.Finish();
    const StagingBufferRef reused = pool.Request(1000, MemoryUsage::Upload);
    REQUIRE((reused.index == first.index || reused.index == second.index));

    // Download buffers come from a separate set.
    const StagingBufferRef download = pool.Request(1000, MemoryUsage::Download);
    REQUIRE(download.index != first.index);
    REQUIRE(download.index != second.index);
}

TEST_CASE("Metal deferred staging buffers wait for FreeDeferred", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    StagingBufferPool pool(*device, scheduler);

    StagingBufferRef deferred = pool.Request(256, MemoryUsage::Download, true);
    scheduler.Finish();
    const StagingBufferRef other = pool.Request(256, MemoryUsage::Download);
    REQUIRE(other.index != deferred.index);

    pool.FreeDeferred(deferred);
    scheduler.Finish();
    scheduler.Finish();
    // Both buffers are idle now; the first idle one in the level is the deferred one.
    const StagingBufferRef reused = pool.Request(256, MemoryUsage::Download);
    REQUIRE(reused.index == deferred.index);
    REQUIRE(pool.GetMemoryUsage() == 512);
}
