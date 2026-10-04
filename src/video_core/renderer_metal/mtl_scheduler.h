// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_scheduler.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "common/common_types.h"

namespace Metal {

class Device;

/**
 * Records GPU work into one command buffer at a time and tracks its completion with ticks, like
 * the Vulkan renderer's Scheduler and MasterSemaphore.
 *
 * Every command buffer is tagged with the tick returned by CurrentTick() while it is recorded.
 * Flush() commits it and moves to the next tick; a completion handler advances KnownGpuTick()
 * when the GPU is done with it. Resources used by tick N can be reused once IsFree(N).
 *
 * Not thread-safe: record and flush from one thread (the GPU thread). KnownGpuTick, IsFree and
 * Wait may be called from any thread.
 */
class Scheduler {
public:
    explicit Scheduler(const Device& device);
    /// Waits for all submitted work to finish.
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// The command buffer for the current tick, created on first use.
    [[nodiscard]] id<MTLCommandBuffer> CommandBuffer();

    /// A blit encoder on the current command buffer, reusing the open one if it is a blit
    /// encoder and ending any other open encoder.
    [[nodiscard]] id<MTLBlitCommandEncoder> BlitEncoder();

    /// A compute encoder on the current command buffer, reusing the open one if it is a compute
    /// encoder and ending any other open encoder.
    [[nodiscard]] id<MTLComputeCommandEncoder> ComputeEncoder();

    /// Ends the open encoder, if any.
    void EndEncoding();

    /// Whether work has been recorded since the last flush.
    [[nodiscard]] bool HasPendingCommands() const noexcept {
        return command_buffer != nil;
    }

    /// Commits the current command buffer and starts the next tick.
    /// @returns The tick that was submitted.
    u64 Flush();

    /// Flushes and waits for the GPU to finish everything submitted so far.
    void Finish();

    /// Waits until the GPU has finished the given tick, flushing first if it is still recording.
    void Wait(u64 tick);

    /// Tick of the command buffer being recorded.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    /// Highest tick the GPU is known to have finished.
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

private:
    enum class EncoderKind {
        None,
        Blit,
        Compute,
    };

    void SignalGpuTick(u64 tick);

    const Device& device;
    id<MTLCommandBuffer> command_buffer;
    id<MTLCommandEncoder> encoder;
    EncoderKind encoder_kind{EncoderKind::None};

    std::atomic<u64> current_tick{1};
    std::atomic<u64> gpu_tick{0};
    std::mutex tick_mutex;
    std::condition_variable tick_cv;
};

} // namespace Metal
