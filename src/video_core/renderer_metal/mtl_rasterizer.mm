// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>

#include "common/alignment.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/device_memory_manager.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Metal {

namespace {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// The clear rectangle: the first scissor when the clear uses it, else everything.
MTLScissorRect ClearRect(const Maxwell& regs) {
    constexpr NSUInteger UNBOUNDED = std::numeric_limits<s32>::max();
    const auto& scissor = regs.scissor_test[0];
    if (!regs.clear_control.use_scissor || !scissor.enable) {
        return {0, 0, UNBOUNDED, UNBOUNDED};
    }
    // Same as the Vulkan renderer: flip the scissor for lower-left window origins.
    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;
    s32 min_y = lower_left ? clip_height - static_cast<s32>(scissor.max_y)
                           : static_cast<s32>(scissor.min_y);
    s32 max_y = lower_left ? clip_height - static_cast<s32>(scissor.min_y)
                           : static_cast<s32>(scissor.max_y);
    min_y = std::max(min_y, 0);
    max_y = std::max(max_y, 0);
    const s32 min_x = static_cast<s32>(scissor.min_x);
    const s32 max_x = static_cast<s32>(scissor.max_x);
    if (max_x <= min_x || max_y <= min_y) {
        return {0, 0, 0, 0};
    }
    return {
        static_cast<NSUInteger>(min_x),
        static_cast<NSUInteger>(min_y),
        static_cast<NSUInteger>(max_x - min_x),
        static_cast<NSUInteger>(max_y - min_y),
    };
}

/// Slices of the attachment a layered clear covers.
std::pair<NSUInteger, NSUInteger> ClearSlices(id<MTLTexture> attachment, u32 base_layer,
                                              u32 layer_count) {
    NSUInteger num_slices = 1;
    if (attachment.textureType == MTLTextureType3D) {
        num_slices = attachment.depth;
    } else if (attachment.textureType == MTLTextureType2DArray ||
               attachment.textureType == MTLTextureType2DMultisampleArray) {
        num_slices = attachment.arrayLength;
    }
    const NSUInteger first = std::min<NSUInteger>(base_layer, num_slices);
    const NSUInteger last = std::min<NSUInteger>(NSUInteger{base_layer} + layer_count, num_slices);
    return {first, last};
}

} // Anonymous namespace

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

void InnerFence::Queue() {
    if (is_stubbed) {
        return;
    }
    wait_tick = scheduler.Flush();
}

bool InnerFence::IsSignaled() const {
    return is_stubbed || scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    if (!is_stubbed) {
        scheduler.Wait(wait_tick);
    }
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, Scheduler& scheduler_)
    : VideoCommon::FenceManager<FenceManagerParams>{rasterizer_, gpu_, texture_cache_,
                                                    buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % VideoCore::Surface::BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

RasterizerMetal::RasterizerMetal(Tegra::GPU& gpu_,
                                 Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                 const Device& device_, Scheduler& scheduler_,
                                 StagingBufferPool& staging_buffer_pool_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_}, scheduler{scheduler_},
      staging_buffer_pool{staging_buffer_pool_}, clear_helper(device, scheduler),
      texture_cache_runtime(device, scheduler, staging_buffer_pool),
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, scheduler, staging_buffer_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      accelerate_dma(buffer_cache, texture_cache),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, scheduler) {}

RasterizerMetal::~RasterizerMetal() {
    Shutdown();
}

void RasterizerMetal::Shutdown() {
    scheduler.Finish();
}

void RasterizerMetal::Draw(bool, u32) {
    if (!logged_draw) {
        logged_draw = true;
        LOG_WARNING(Render_Metal, "Draws are not implemented on Metal yet; skipping them");
    }
}

void RasterizerMetal::DrawTexture() {
    Draw(false, 1);
}

void RasterizerMetal::DispatchCompute() {
    if (!logged_compute) {
        logged_compute = true;
        LOG_WARNING(Render_Metal, "Compute dispatches are not implemented on Metal yet");
    }
}

void RasterizerMetal::Clear(u32 layer_count) {
    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G ||
                           regs.clear_surface.B || regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }
    gpu_memory->FlushCaching();

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    MTLScissorRect rect = ClearRect(regs);
    rect.width = std::min<NSUInteger>(rect.width, framebuffer->Width());
    rect.height = std::min<NSUInteger>(rect.height, framebuffer->Height());
    if (rect.width == 0 || rect.height == 0) {
        return;
    }
    const u32 base_layer = regs.clear_surface.layer;

    const u32 color_index = regs.clear_surface.RT;
    id<MTLTexture> color = color_index < NUM_RT ? framebuffer->ColorAttachment(color_index) : nil;
    if (use_color && color != nil) {
        using namespace VideoCore::Surface;
        const PixelFormat format = PixelFormatFromRenderTargetFormat(regs.rt[color_index].format);
        const bool is_integer = IsPixelFormatInteger(format);
        const bool is_signed = IsPixelFormatSignedInteger(format);
        const size_t int_size = PixelComponentSizeBitsInteger(format);
        std::array<double, 4> value{};
        ClearHelper::ColorKind kind = ClearHelper::ColorKind::Float;
        for (size_t i = 0; i < 4; ++i) {
            // Integer conversions match the Vulkan renderer.
            if (!is_integer) {
                value[i] = regs.clear_color[i];
            } else if (!is_signed) {
                kind = ClearHelper::ColorKind::Uint;
                value[i] = static_cast<double>(static_cast<u32>(
                    static_cast<f32>(static_cast<u64>(int_size) << 1U) * regs.clear_color[i]));
            } else {
                kind = ClearHelper::ColorKind::Sint;
                value[i] = static_cast<double>(static_cast<s32>(
                    static_cast<f32>(static_cast<s64>(int_size - 1) << 1) *
                    (regs.clear_color[i] - 0.5f)));
            }
        }
        const u8 mask = static_cast<u8>(regs.clear_surface.R | regs.clear_surface.G << 1 |
                                        regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
        const auto [first, last] = ClearSlices(color, base_layer, layer_count);
        for (NSUInteger slice = first; slice < last; ++slice) {
            clear_helper.ClearColor(color, slice, rect, mask, kind, value);
        }
    }

    id<MTLTexture> depth = framebuffer->DepthAttachment();
    if ((use_depth || use_stencil) && depth != nil) {
        const auto [first, last] = ClearSlices(depth, base_layer, layer_count);
        for (NSUInteger slice = first; slice < last; ++slice) {
            clear_helper.ClearDepthStencil(
                depth, slice, rect, use_depth && framebuffer->HasAspectDepthBit(),
                regs.clear_depth, use_stencil && framebuffer->HasAspectStencilBit(),
                static_cast<u8>(regs.clear_stencil), static_cast<u8>(regs.stencil_front_mask));
        }
    }
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType) {}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32) {
    // Like the null renderer: report the payload without counting anything.
    if (!gpu_memory) {
        return;
    }
    if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
        gpu_memory->Write<u64>(gpu_addr + 8, gpu.GetTicks());
        gpu_memory->Write<u64>(gpu_addr, static_cast<u64>(payload));
    } else {
        gpu_memory->Write<u32>(gpu_addr, payload);
    }
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelNormal()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        if (const auto area = texture_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    return {
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
}

void RasterizerMetal::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return false;
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.WriteMemory(addr, size);
    return false;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.WriteMemory(addr, size);
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.WriteMemory(addr, size);
}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerMetal::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerMetal::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    if (Settings::IsGPULevelExtreme()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerMetal::WaitForIdle() {
    // Metal orders work on one queue and tracks hazards between encoders, so there is nothing to
    // wait on inside a command buffer.
    fence_manager.SignalOrdering();
}

void RasterizerMetal::FragmentBarrier() {}

void RasterizerMetal::TiledCacheBarrier() {}

void RasterizerMetal::FlushCommands() {
    if (scheduler.HasPendingCommands()) {
        scheduler.Flush();
    }
}

void RasterizerMetal::TickFrame() {
    fence_manager.TickFrame();
    staging_buffer_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.TickFrame();
}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock<std::recursive_mutex> lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.WriteMemory(*cpu_addr, copy_size);
}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    // The caches track render target, vertex buffer and similar changes through these tables.
    VideoCommon::Dirty::SetupDirtyFlags(channel.maxwell_3d->dirty.tables);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    channel.maxwell_3d->dirty.flags.set();
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.EraseChannel(channel_id);
    buffer_cache.EraseChannel(channel_id);
}

u64 RasterizerMetal::GetTotalVram() const {
    return device.GetRecommendedWorkingSetSize();
}

u64 RasterizerMetal::GetUsedVram() const {
    return device.GetCurrentAllocatedSize();
}

u64 RasterizerMetal::GetBufferMemoryUsage() const {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.GetBufferVRAMStats().total_used_bytes;
}

u64 RasterizerMetal::GetTextureMemoryUsage() const {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.GetVRAMStats().total_used_bytes;
}

u64 RasterizerMetal::GetStagingMemoryUsage() const {
    return staging_buffer_pool.GetMemoryUsage();
}

std::optional<FramebufferTexture> RasterizerMetal::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr) {
    if (framebuffer_addr == 0) {
        return std::nullopt;
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (image_view == nullptr) {
        return std::nullopt;
    }
    id<MTLTexture> texture = image_view->Handle(Shader::TextureType::Color2D);
    if (texture == nil) {
        return std::nullopt;
    }
    return FramebufferTexture{
        .texture = texture,
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
}

} // namespace Metal
