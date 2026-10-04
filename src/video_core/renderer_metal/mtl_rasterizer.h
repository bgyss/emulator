// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_rasterizer.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <memory>
#include <optional>
#include <unordered_map>

#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/fence_manager.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_clear_helper.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace Tegra {
struct FramebufferConfig;
}

namespace Metal {

class Device;
class Scheduler;
class StagingBufferPool;

/// Stands in for a query cache until Metal implements queries; queries are written right away.
class QueryCache {
public:
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const {
        return false;
    }
    [[nodiscard]] bool HasUncommittedFlushes() const {
        return false;
    }
    void CommitAsyncFlushes() {}
    void PopAsyncFlushes() {}
};

class InnerFence : public VideoCommon::FenceBase {
public:
    explicit InnerFence(Scheduler& scheduler_, bool is_stubbed_);

    void Queue();
    [[nodiscard]] bool IsSignaled() const;
    void Wait();

private:
    Scheduler& scheduler;
    u64 wait_tick{};
};
using Fence = std::shared_ptr<InnerFence>;

struct FenceManagerParams {
    using FenceType = Fence;
    using BufferCacheType = BufferCache;
    using TextureCacheType = TextureCache;
    using QueryCacheType = QueryCache;

    static constexpr bool HAS_ASYNC_CHECK = true;
};

class FenceManager final : public VideoCommon::FenceManager<FenceManagerParams> {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
                          TextureCache& texture_cache, BufferCache& buffer_cache,
                          QueryCache& query_cache, Scheduler& scheduler);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;

private:
    Scheduler& scheduler;
};

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache);

    bool BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& buffer_operand,
                            const Tegra::DMA::ImageOperand& image_operand);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
};

/// A guest framebuffer the GPU rendered, found in the texture cache.
struct FramebufferTexture {
    id<MTLTexture> texture;
    u32 width{};
    u32 height{};
};

/**
 * Rasterizer for the Metal renderer.
 *
 * Owns the buffer, texture and pipeline caches and keeps them in sync with guest memory, draws,
 * executes clears, copies and blits, and finds GPU-rendered framebuffers for presentation.
 * Compute dispatches, indirect draws and transform feedback are not implemented yet.
 */
class RasterizerMetal final : public VideoCore::RasterizerInterface,
                              protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerMetal(Tegra::GPU& gpu, Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device, Scheduler& scheduler,
                             StagingBufferPool& staging_buffer_pool);
    ~RasterizerMetal() override;

    void Shutdown() override;
    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;

    u64 GetTotalVram() const override;
    u64 GetUsedVram() const override;
    u64 GetBufferMemoryUsage() const override;
    u64 GetTextureMemoryUsage() const override;
    u64 GetStagingMemoryUsage() const override;

    /// The texture the GPU rendered the framebuffer into, if the texture cache has one.
    std::optional<FramebufferTexture> AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                        DAddr framebuffer_addr);

private:
    /// Sets the render encoder state the pipeline doesn't hold from the Maxwell registers.
    /// @returns false when the draw can't produce anything and should be skipped.
    bool UpdateDynamicState(id<MTLRenderCommandEncoder> encoder, const Framebuffer& framebuffer);

    id<MTLDepthStencilState> DepthStencilState(const Framebuffer& framebuffer);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;

    ClearHelper clear_helper;
    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    QueryCache query_cache;
    PipelineCache pipeline_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;
    /// Sets up the Maxwell dirty tables the pipeline key and the caches read. It only tracks
    /// registers, so it is shared with the Vulkan renderer.
    Vulkan::StateTracker state_tracker;

    std::unordered_map<u64, id<MTLDepthStencilState>> depth_stencil_states;
    u32 draw_counter{};
    bool logged_topology{};
    bool logged_compute{};
};

} // namespace Metal
