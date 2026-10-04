// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace Metal {
namespace {

constexpr size_t NULL_BUFFER_SIZE = 64 * 1024;

constexpr std::array<u32, 6> QUAD_SWIZZLE{0, 1, 2, 0, 2, 3};
constexpr std::array<u32, 6> QUAD_STRIP_SWIZZLE{0, 3, 1, 0, 2, 3};

// Index rewriting kernels. Sources are read a byte at a time so an index buffer can start at any
// byte offset; the bound offset is aligned down to 4 bytes and the rest is passed in byte_offset.
constexpr const char* INDEX_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct IndexParams {
    uint byte_offset;
    uint index_shift; // 0: u8, 1: u16, 2: u32
    uint count;       // indices for widen_u8, quads for assemble_quads
    uint is_strip;
};

constant uint quad_swizzle[6] = {0, 1, 2, 0, 2, 3};
constant uint quad_strip_swizzle[6] = {0, 3, 1, 0, 2, 3};

static uint read_index(device const uchar* src, uint byte_offset, uint shift, uint i) {
    device const uchar* p = src + byte_offset + (i << shift);
    switch (shift) {
    case 0:
        return p[0];
    case 1:
        return uint(p[0]) | (uint(p[1]) << 8);
    default:
        return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    }
}

kernel void widen_u8(device const uchar* src [[buffer(0)]], device ushort* dst [[buffer(1)]],
                     constant IndexParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    // Most primitive restart indices are 0xFF; keep them restart indices at 16 bits.
    const uint index = src[p.byte_offset + i];
    dst[i] = ushort(index == 0xFF ? 0xFFFF : index);
}

kernel void assemble_quads(device const uchar* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                           constant IndexParams& p [[buffer(2)]],
                           uint quad [[thread_position_in_grid]]) {
    if (quad >= p.count) {
        return;
    }
    for (uint vertex = 0; vertex < 6; ++vertex) {
        const uint i = p.is_strip != 0 ? quad * 2 + quad_strip_swizzle[vertex]
                                       : quad * 4 + quad_swizzle[vertex];
        dst[quad * 6 + vertex] = read_index(src, p.byte_offset, p.index_shift, i);
    }
}
)";

struct IndexParams {
    u32 byte_offset;
    u32 index_shift;
    u32 count;
    u32 is_strip;
};

u32 IndexShift(Tegra::Engines::Maxwell3D::Regs::IndexFormat format) {
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;
    switch (format) {
    case IndexFormat::UnsignedByte:
        return 0;
    case IndexFormat::UnsignedShort:
        return 1;
    case IndexFormat::UnsignedInt:
        return 2;
    }
    ASSERT_MSG(false, "Invalid index format={}", static_cast<u32>(format));
    return 2;
}

u32 NumQuads(bool is_strip, u32 count) {
    if (is_strip) {
        return count >= 4 ? (count - 2) / 2 : 0;
    }
    return count / 4;
}

} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime&, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase(null_params), tracker{4096} {}

Buffer::Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), tracker{SizeBytes()} {
    buffer = [runtime.device.GetDevice() newBufferWithLength:SizeBytes()
                                                     options:MTLResourceStorageModePrivate];
    if (buffer == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to allocate a {} byte buffer", SizeBytes());
        throw std::runtime_error("Failed to allocate a Metal buffer");
    }
    buffer.label = [NSString
        stringWithFormat:@"Buffer 0x%llx", static_cast<unsigned long long>(CpuAddr())];
}

BufferCacheRuntime::BufferCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                       StagingBufferPool& staging_pool_)
    : device{device_}, scheduler{scheduler_}, staging_pool{staging_pool_} {}

BufferCacheRuntime::~BufferCacheRuntime() = default;

void BufferCacheRuntime::TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept {
    for (auto it = slot_buffers.begin(); it != slot_buffers.end(); it++) {
        it->ResetUsageTracking();
    }
}

void BufferCacheRuntime::Finish() {
    scheduler.Finish();
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return device.GetRecommendedWorkingSetSize();
}

u64 BufferCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetCurrentAllocatedSize();
}

u32 BufferCacheRuntime::GetStorageBufferAlignment() const {
    // Device-address-space buffer offsets need 16 bytes on every Apple GPU family, which is
    // also what MoltenVK reports as minStorageBufferOffsetAlignment.
    return 16;
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, MemoryUsage::Upload);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Download, deferred);
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

void BufferCacheRuntime::CopyBuffer(id<MTLBuffer> dst_buffer, id<MTLBuffer> src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies,
                                    [[maybe_unused]] bool barrier,
                                    [[maybe_unused]] bool can_reorder_upload) {
    if (dst_buffer == nil || src_buffer == nil) {
        return;
    }
    id<MTLBlitCommandEncoder> encoder = nil;
    for (const VideoCommon::BufferCopy& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        if (encoder == nil) {
            encoder = scheduler.BlitEncoder();
        }
        [encoder copyFromBuffer:src_buffer
                   sourceOffset:copy.src_offset
                       toBuffer:dst_buffer
              destinationOffset:copy.dst_offset
                           size:copy.size];
    }
}

void BufferCacheRuntime::ClearBuffer(id<MTLBuffer> dst_buffer, u32 offset, size_t size,
                                     u32 value) {
    if (dst_buffer == nil || size == 0) {
        return;
    }
    const u8 low_byte = static_cast<u8>(value);
    if (value == low_byte * 0x01010101U) {
        // fillBuffer writes a single byte value, which covers 0 and other repeated bytes.
        [scheduler.BlitEncoder() fillBuffer:dst_buffer range:NSMakeRange(offset, size)
                                      value:low_byte];
        return;
    }
    // Any other pattern is written through a staging buffer.
    const StagingBufferRef staging = staging_pool.Request(size, MemoryUsage::Upload);
    u8* const data = staging.mapped_span.data();
    for (size_t i = 0; i < size; i += sizeof(u32)) {
        std::memcpy(data + i, &value, std::min(sizeof(u32), size - i));
    }
    [scheduler.BlitEncoder() copyFromBuffer:staging.buffer
                               sourceOffset:staging.offset
                                   toBuffer:dst_buffer
                          destinationOffset:offset
                                       size:size];
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 first, u32 count, id<MTLBuffer> buffer, u32 offset,
                                         [[maybe_unused]] u32 size) {
    if (buffer == nil) {
        index_buffer = {
            .buffer = NullBuffer(),
            .offset = 0,
            .type = MTLIndexTypeUInt32,
            .rewritten_count = std::nullopt,
        };
        return;
    }
    const bool is_quads = topology == PrimitiveTopology::Quads;
    const bool is_strip = topology == PrimitiveTopology::QuadStrip;
    if (!is_quads && !is_strip && index_format != IndexFormat::UnsignedByte) {
        index_buffer = {
            .buffer = buffer,
            .offset = offset,
            .type = index_format == IndexFormat::UnsignedShort ? MTLIndexTypeUInt16
                                                               : MTLIndexTypeUInt32,
            .rewritten_count = std::nullopt,
        };
        return;
    }

    const u32 index_shift = IndexShift(index_format);
    const bool assemble_quads = is_quads || is_strip;
    const u32 num_quads = assemble_quads ? NumQuads(is_strip, count) : 0;
    const u32 out_count = assemble_quads ? num_quads * 6 : count;
    const u32 threads = assemble_quads ? num_quads : count;
    if (out_count == 0) {
        index_buffer = {
            .buffer = NullBuffer(),
            .offset = 0,
            .type = MTLIndexTypeUInt32,
            .rewritten_count = 0,
        };
        return;
    }
    const size_t out_index_size = assemble_quads ? sizeof(u32) : sizeof(u16);
    const StagingBufferRef out =
        staging_pool.Request(size_t{out_count} * out_index_size, MemoryUsage::Upload);

    const u32 src_offset = offset + (first << index_shift);
    const u32 aligned_src_offset = src_offset & ~3U;
    const IndexParams params{
        .byte_offset = src_offset - aligned_src_offset,
        .index_shift = index_shift,
        .count = threads,
        .is_strip = is_strip ? 1U : 0U,
    };
    id<MTLComputePipelineState> pipeline = IndexPipeline(assemble_quads);
    id<MTLComputeCommandEncoder> encoder = scheduler.ComputeEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:aligned_src_offset atIndex:0];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger width =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, NSUInteger{256});
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];

    index_buffer = {
        .buffer = out.buffer,
        .offset = static_cast<u32>(out.offset),
        .type = assemble_quads ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16,
        .rewritten_count = out_count,
    };
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    const bool is_strip = topology == PrimitiveTopology::QuadStrip;
    const u32 num_quads = NumQuads(is_strip, count);
    if (num_quads == 0) {
        index_buffer = {
            .buffer = NullBuffer(),
            .offset = 0,
            .type = MTLIndexTypeUInt32,
            .rewritten_count = 0,
        };
        return;
    }
    // Non-indexed quads only need vertex numbers, so build the triangle list on the CPU.
    const u32 out_count = num_quads * 6;
    const StagingBufferRef out =
        staging_pool.Request(size_t{out_count} * sizeof(u32), MemoryUsage::Upload);
    const auto& swizzle = is_strip ? QUAD_STRIP_SWIZZLE : QUAD_SWIZZLE;
    const u32 quad_step = is_strip ? 2 : 4;
    u8* data = out.mapped_span.data();
    for (u32 quad = 0; quad < num_quads; ++quad) {
        std::array<u32, 6> indices;
        for (size_t vertex = 0; vertex < indices.size(); ++vertex) {
            indices[vertex] = first + quad * quad_step + swizzle[vertex];
        }
        std::memcpy(data, indices.data(), sizeof(indices));
        data += sizeof(indices);
    }
    index_buffer = {
        .buffer = out.buffer,
        .offset = static_cast<u32>(out.offset),
        .type = MTLIndexTypeUInt32,
        .rewritten_count = out_count,
    };
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, id<MTLBuffer> buffer, u32 offset, u32 size,
                                          u32 stride) {
    if (index >= vertex_buffers.size()) {
        return;
    }
    if (buffer == nil) {
        buffer = NullBuffer();
        offset = 0;
        size = 0;
    }
    vertex_buffers[index] = {
        .buffer = buffer,
        .offset = offset,
        .size = size,
        .stride = stride,
    };
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        BindVertexBuffer(bindings.min_index + i, bindings.buffers[i]->Handle(),
                         static_cast<u32>(bindings.offsets[i]),
                         static_cast<u32>(bindings.sizes[i]),
                         static_cast<u32>(bindings.strides[i]));
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffer(u32, id<MTLBuffer>, u32, u32) {
    if (!logged_transform_feedback) {
        LOG_WARNING(Render_Metal, "Transform feedback is not supported on Metal");
        logged_transform_feedback = true;
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>&) {
    BindTransformFeedbackBuffer(0, nil, 0, 0);
}

void BufferCacheRuntime::AddDescriptor(id<MTLBuffer> buffer, u32 offset, u32 size,
                                       BufferDescriptor::Kind kind, bool is_written,
                                       VideoCore::Surface::PixelFormat format) {
    descriptors.push_back({
        .buffer = buffer,
        .offset = offset,
        .size = size,
        .kind = kind,
        .is_written = is_written,
        .format = format,
    });
}

id<MTLBuffer> BufferCacheRuntime::NullBuffer() {
    if (null_buffer == nil) {
        null_buffer = [device.GetDevice() newBufferWithLength:NULL_BUFFER_SIZE
                                                      options:MTLResourceStorageModeShared];
        std::memset(null_buffer.contents, 0, NULL_BUFFER_SIZE);
        null_buffer.label = @"null buffer";
    }
    return null_buffer;
}

id<MTLComputePipelineState> BufferCacheRuntime::IndexPipeline(bool assemble_quads) {
    __strong id<MTLComputePipelineState>& pipeline =
        assemble_quads ? assemble_quads_pipeline : widen_u8_pipeline;
    if (pipeline != nil) {
        return pipeline;
    }
    NSError* error = nil;
    if (index_library == nil) {
        index_library = [device.GetDevice() newLibraryWithSource:@(INDEX_SHADER_SOURCE)
                                                         options:nil
                                                           error:&error];
        if (index_library == nil) {
            throw std::runtime_error(std::string{"Failed to compile Metal index shaders: "} +
                                     error.localizedDescription.UTF8String);
        }
    }
    id<MTLFunction> function =
        [index_library newFunctionWithName:assemble_quads ? @"assemble_quads" : @"widen_u8"];
    pipeline = [device.GetDevice() newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string{"Failed to create Metal index pipeline: "} +
                                 error.localizedDescription.UTF8String);
    }
    return pipeline;
}

} // namespace Metal
