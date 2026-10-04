// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace Metal {

Scheduler::Scheduler(const Device& device_) : device{device_} {}

Scheduler::~Scheduler() {
    Finish();
}

id<MTLCommandBuffer> Scheduler::CommandBuffer() {
    if (command_buffer == nil) {
        command_buffer = [device.GetQueue() commandBuffer];
        command_buffer.label = [NSString stringWithFormat:@"tick %llu", CurrentTick()];
    }
    return command_buffer;
}

id<MTLBlitCommandEncoder> Scheduler::BlitEncoder() {
    if (encoder_kind != EncoderKind::Blit) {
        EndEncoding();
        encoder = [CommandBuffer() blitCommandEncoder];
        encoder_kind = EncoderKind::Blit;
    }
    return (id<MTLBlitCommandEncoder>)encoder;
}

id<MTLComputeCommandEncoder> Scheduler::ComputeEncoder() {
    if (encoder_kind != EncoderKind::Compute) {
        EndEncoding();
        encoder = [CommandBuffer() computeCommandEncoder];
        encoder_kind = EncoderKind::Compute;
    }
    return (id<MTLComputeCommandEncoder>)encoder;
}

void Scheduler::EndEncoding() {
    if (encoder != nil) {
        [encoder endEncoding];
        encoder = nil;
    }
    encoder_kind = EncoderKind::None;
}

u64 Scheduler::Flush() {
    EndEncoding();
    // Commit even an empty command buffer so every tick completes and waiters wake up.
    id<MTLCommandBuffer> submitted = CommandBuffer();
    const u64 tick = CurrentTick();
    [submitted addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status == MTLCommandBufferStatusError) {
          LOG_ERROR(Render_Metal, "Command buffer for tick {} failed: {}", tick,
                    completed.error.localizedDescription.UTF8String);
      }
      SignalGpuTick(tick);
    }];
    [submitted commit];
    command_buffer = nil;
    current_tick.fetch_add(1, std::memory_order_release);
    return tick;
}

void Scheduler::Finish() {
    Wait(Flush());
}

void Scheduler::Wait(u64 tick) {
    if (tick >= CurrentTick()) {
        // Ticks after the one being recorded will never be signaled; wait for what exists.
        tick = Flush();
    }
    if (IsFree(tick)) {
        return;
    }
    std::unique_lock lock{tick_mutex};
    tick_cv.wait(lock, [this, tick] { return IsFree(tick); });
}

void Scheduler::SignalGpuTick(u64 tick) {
    {
        std::scoped_lock lock{tick_mutex};
        // Command buffers on one queue complete in order, but keep the maximum to be safe.
        u64 known = gpu_tick.load(std::memory_order_relaxed);
        while (known < tick &&
               !gpu_tick.compare_exchange_weak(known, tick, std::memory_order_release)) {
        }
    }
    tick_cv.notify_all();
}

} // namespace Metal
