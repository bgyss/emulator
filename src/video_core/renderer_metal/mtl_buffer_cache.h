// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_buffer_cache.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/buffer_cache/usage_tracker.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/surface.h"

namespace Metal {

class Device;
class Scheduler;

class BufferCacheRuntime;

/// A guest buffer backed by a private-storage MTLBuffer.
class Buffer : public VideoCommon::BufferBase {
public:
    /// A null buffer has no storage; its handle is nil and copies to or from it are skipped.
    explicit Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params);
    explicit Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_);

    [[nodiscard]] id<MTLBuffer> Handle() const noexcept {
        return buffer;
    }

    [[nodiscard]] bool IsRegionUsed(u64 offset, u64 size) const noexcept {
        return tracker.IsUsed(offset, size);
    }

    void MarkUsage(u64 offset, u64 size) noexcept {
        tracker.Track(offset, size);
    }

    void ResetUsageTracking() noexcept {
        tracker.Reset();
    }

    operator id<MTLBuffer>() const noexcept {
        return buffer;
    }

private:
    id<MTLBuffer> buffer;
    VideoCommon::UsageTracker tracker;
};

/// The index buffer the next indexed draw reads.
struct IndexBufferBinding {
    id<MTLBuffer> buffer;
    /// Offset of index 0. When the indices are bound as-is, the draw adds its first index.
    u32 offset{};
    MTLIndexType type{MTLIndexTypeUInt32};
    /// Set when the runtime rewrote the guest indices (8-bit to 16-bit, quads to triangles).
    /// The rewritten indices start at the draw's first index, so the draw must start at index 0
    /// and draw this many indices.
    std::optional<u32> rewritten_count;
};

struct VertexBufferBinding {
    id<MTLBuffer> buffer;
    u32 offset{};
    u32 size{};
    u32 stride{};
};

/// A buffer the next pipeline binds, in the order the buffer cache visits its descriptors:
/// uniform buffers, then storage buffers, then texture buffers, stage by stage.
struct BufferDescriptor {
    enum class Kind {
        Uniform,
        Storage,
        Texture,
    };

    id<MTLBuffer> buffer;
    u32 offset{};
    u32 size{};
    Kind kind{};
    bool is_written{};
    /// Texel format, for Kind::Texture only.
    VideoCore::Surface::PixelFormat format{};
};

/**
 * Backend half of VideoCommon::BufferCache for Metal.
 *
 * Copies and clears are recorded on the scheduler's blit encoder. Metal tracks hazards between
 * encoders for the buffers created here, so the barrier hooks are no-ops.
 *
 * Metal binds buffers when a draw is encoded rather than ahead of it, so the Bind* calls only
 * record bindings; the rasterizer reads them back when it encodes the draw. Index formats and
 * topologies Metal can't draw (8-bit indices, quads, quad strips) are rewritten here.
 */
class BufferCacheRuntime {
    friend Buffer;

    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

public:
    explicit BufferCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                StagingBufferPool& staging_pool_);
    ~BufferCacheRuntime();

    BufferCacheRuntime(const BufferCacheRuntime&) = delete;
    BufferCacheRuntime& operator=(const BufferCacheRuntime&) = delete;

    void TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept;

    void Finish();

    u64 GetDeviceLocalMemory() const;

    u64 GetDeviceMemoryUsage() const;

    void CleanupUnusedBuffers() {}

    bool CanReportMemoryUsage() const {
        return true;
    }

    u32 GetStorageBufferAlignment() const;

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size);

    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    /// There is no separate upload command buffer to reorder into.
    bool CanReorderUpload(const Buffer&, std::span<const VideoCommon::BufferCopy>) {
        return false;
    }

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void PreCopyBarrier() {}

    void CopyBuffer(id<MTLBuffer> dst_buffer, id<MTLBuffer> src_buffer,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder_upload = false);

    void PostCopyBarrier() {}

    void ClearBuffer(id<MTLBuffer> dst_buffer, u32 offset, size_t size, u32 value);

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 first, u32 count,
                         id<MTLBuffer> buffer, u32 offset, u32 size);

    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    void BindVertexBuffer(u32 index, id<MTLBuffer> buffer, u32 offset, u32 size, u32 stride);

    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    void BindTransformFeedbackBuffer(u32 index, id<MTLBuffer> buffer, u32 offset, u32 size);

    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    std::span<u8> BindMappedUniformBuffer([[maybe_unused]] size_t stage,
                                          [[maybe_unused]] u32 binding_index, u32 size) {
        const StagingBufferRef ref = staging_pool.Request(size, MemoryUsage::Upload);
        AddDescriptor(ref.buffer, static_cast<u32>(ref.offset), size,
                      BufferDescriptor::Kind::Uniform);
        return ref.mapped_span;
    }

    /// Metal reads vertex buffers without bounds checks, so short ones are padded like on
    /// MoltenVK.
    bool NeedsPaddedVertexBuffers() const {
        return true;
    }

    std::span<u8> BindMappedVertexBuffer(u32 index, u32 size, u32 stride) {
        const StagingBufferRef ref = staging_pool.Request(size, MemoryUsage::Upload);
        BindVertexBuffer(index, ref.buffer, static_cast<u32>(ref.offset), size, stride);
        return ref.mapped_span;
    }

    void BindUniformBuffer(id<MTLBuffer> buffer, u32 offset, u32 size) {
        AddDescriptor(buffer, offset, size, BufferDescriptor::Kind::Uniform);
    }

    void BindStorageBuffer(id<MTLBuffer> buffer, u32 offset, u32 size, bool is_written) {
        AddDescriptor(buffer, offset, size, BufferDescriptor::Kind::Storage, is_written);
    }

    void BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                           VideoCore::Surface::PixelFormat format) {
        AddDescriptor(buffer.Handle(), offset, size, BufferDescriptor::Kind::Texture, false,
                      format);
    }

    [[nodiscard]] const IndexBufferBinding& GetIndexBuffer() const noexcept {
        return index_buffer;
    }

    [[nodiscard]] std::span<const VertexBufferBinding> GetVertexBuffers() const noexcept {
        return vertex_buffers;
    }

    /// Descriptors recorded since the last ClearDescriptors, in binding order.
    [[nodiscard]] std::span<const BufferDescriptor> GetDescriptors() const noexcept {
        return descriptors;
    }

    void ClearDescriptors() noexcept {
        descriptors.clear();
    }

private:
    void AddDescriptor(id<MTLBuffer> buffer, u32 offset, u32 size, BufferDescriptor::Kind kind,
                       bool is_written = false, VideoCore::Surface::PixelFormat format = {});

    /// A zero-filled buffer bound in place of a missing index or vertex buffer.
    id<MTLBuffer> NullBuffer();

    id<MTLComputePipelineState> IndexPipeline(bool assemble_quads);

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;

    id<MTLBuffer> null_buffer;
    id<MTLLibrary> index_library;
    id<MTLComputePipelineState> widen_u8_pipeline;
    id<MTLComputePipelineState> assemble_quads_pipeline;

    IndexBufferBinding index_buffer;
    std::array<VertexBufferBinding, VideoCommon::NUM_VERTEX_BUFFERS> vertex_buffers{};
    std::vector<BufferDescriptor> descriptors;
    bool logged_transform_feedback{};
};

struct BufferCacheParams {
    using Runtime = Metal::BufferCacheRuntime;
    using Buffer = Metal::Buffer;
    using Async_Buffer = Metal::StagingBufferRef;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace Metal
