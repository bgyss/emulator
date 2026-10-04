// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>

#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/device_memory_manager.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/draw_manager.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/maxwell_to_mtl.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Metal {

namespace {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Draws recorded before the command buffer is committed, so the GPU starts on long frames.
constexpr u32 DRAWS_PER_FLUSH = 2048;

/// The first viewport, computed like the Vulkan renderer does. With the vertex shader's Y flip,
/// Metal maps it to the same framebuffer coordinates as Vulkan, negative heights included.
MTLViewport ViewportState(const Maxwell& regs) {
    if (!regs.viewport_scale_offset_enabled) {
        const auto width = static_cast<double>(regs.surface_clip.width);
        const auto height = static_cast<double>(regs.surface_clip.height);
        return {
            .originX = static_cast<double>(regs.surface_clip.x),
            .originY = static_cast<double>(regs.surface_clip.y),
            .width = width > 0.0 ? width : 1.0,
            .height = height > 0.0 ? height : 1.0,
            .znear = 0.0,
            .zfar = 1.0,
        };
    }
    const auto& src = regs.viewport_transform[0];
    const double x = src.translate_x - src.scale_x;
    const double width = src.scale_x * 2.0f;
    double y = src.translate_y - src.scale_y;
    double height = src.scale_y * 2.0f;
    if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
        y += regs.surface_clip.height;
        height = -height;
    }
    if (src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
        y += height;
        height = -height;
    }
    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    return {
        .originX = x,
        .originY = y,
        .width = width != 0.0 ? width : 1.0,
        .height = height != 0.0 ? height : 1.0,
        .znear = std::clamp(src.translate_z - src.scale_z * reduce_z, 0.0f, 1.0f),
        .zfar = std::clamp(src.translate_z + src.scale_z, 0.0f, 1.0f),
    };
}

/// The first scissor, clipped to the render area; the whole area when scissoring is off.
MTLScissorRect ScissorState(const Maxwell& regs, u32 width, u32 height) {
    const auto& src = regs.scissor_test[0];
    if (!src.enable) {
        return {0, 0, width, height};
    }
    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;
    const s32 min_y = std::max(lower_left ? clip_height - static_cast<s32>(src.max_y)
                                          : static_cast<s32>(src.min_y),
                               0);
    const s32 max_y = std::max(lower_left ? clip_height - static_cast<s32>(src.min_y)
                                          : static_cast<s32>(src.max_y),
                               0);
    const u32 x0 = std::min<u32>(src.min_x, width);
    const u32 x1 = std::clamp<u32>(src.max_x, x0, width);
    const u32 y0 = std::min<u32>(static_cast<u32>(min_y), height);
    const u32 y1 = std::clamp<u32>(static_cast<u32>(max_y), y0, height);
    return {x0, y0, x1 - x0, y1 - y0};
}

MTLStencilDescriptor* StencilFace(Maxwell::StencilOp::Op fail, Maxwell::StencilOp::Op zfail,
                                  Maxwell::StencilOp::Op zpass, Maxwell::ComparisonOp func,
                                  u32 read_mask, u32 write_mask) {
    MTLStencilDescriptor* face = [[MTLStencilDescriptor alloc] init];
    face.stencilFailureOperation = MaxwellToMTL::StencilOp(fail);
    face.depthFailureOperation = MaxwellToMTL::StencilOp(zfail);
    face.depthStencilPassOperation = MaxwellToMTL::StencilOp(zpass);
    face.stencilCompareFunction = MaxwellToMTL::ComparisonOp(func);
    face.readMask = read_mask & 0xFF;
    face.writeMask = write_mask & 0xFF;
    return face;
}

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
      pipeline_cache(device_memory, device, scheduler, buffer_cache, buffer_cache_runtime,
                     texture_cache),
      accelerate_dma(buffer_cache, texture_cache),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, scheduler) {}

RasterizerMetal::~RasterizerMetal() {
    Shutdown();
}

void RasterizerMetal::Shutdown() {
    scheduler.Finish();
}

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    SCOPE_EXIT {
        gpu.TickWork();
    };
    gpu_memory->FlushCaching();

    if (instance_count == 0) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (pipeline == nullptr) {
        return;
    }
    const auto& draw_state = maxwell3d->draw_manager->GetDrawState();
    const auto primitive = MaxwellToMTL::PrimitiveType(draw_state.topology);
    if (!primitive) {
        if (!logged_topology) {
            logged_topology = true;
            LOG_WARNING(Render_Metal, "Topology {} is not implemented on Metal; skipping its draws",
                        static_cast<u32>(draw_state.topology));
        }
        return;
    }

    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    id<MTLRenderCommandEncoder> encoder = pipeline->Configure(*maxwell3d, *gpu_memory, is_indexed);
    if (encoder == nil) {
        return;
    }
    if (!UpdateDynamicState(encoder, *texture_cache.GetFramebuffer())) {
        return;
    }
    const bool is_quads = draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
                          draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip;
    if (is_indexed || is_quads) {
        const IndexBufferBinding& index = buffer_cache_runtime.GetIndexBuffer();
        NSUInteger offset = index.offset;
        u32 count = 0;
        if (index.rewritten_count) {
            // The rewritten indices start at the draw's first index.
            count = *index.rewritten_count;
        } else {
            count = draw_state.index_buffer.count;
            const NSUInteger index_size = index.type == MTLIndexTypeUInt16 ? 2 : 4;
            offset += NSUInteger{draw_state.index_buffer.first} * index_size;
        }
        if (count == 0 || index.buffer == nil) {
            return;
        }
        // Non-indexed quads were rewritten into absolute vertex numbers.
        const NSInteger base_vertex = is_indexed ? static_cast<s32>(draw_state.base_index) : 0;
        [encoder drawIndexedPrimitives:*primitive
                            indexCount:count
                             indexType:index.type
                           indexBuffer:index.buffer
                     indexBufferOffset:offset
                         instanceCount:instance_count
                            baseVertex:base_vertex
                          baseInstance:draw_state.base_instance];
    } else {
        if (draw_state.vertex_buffer.count == 0) {
            return;
        }
        [encoder drawPrimitives:*primitive
                    vertexStart:draw_state.vertex_buffer.first
                    vertexCount:draw_state.vertex_buffer.count
                  instanceCount:instance_count
                   baseInstance:draw_state.base_instance];
    }
    if (++draw_counter >= DRAWS_PER_FLUSH) {
        draw_counter = 0;
        scheduler.Flush();
    }
}

void RasterizerMetal::DrawTexture() {
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_WARNING(Render_Metal, "Texture draws are not implemented on Metal yet");
    }
}

bool RasterizerMetal::UpdateDynamicState(id<MTLRenderCommandEncoder> encoder,
                                         const Framebuffer& framebuffer) {
    const auto& regs = maxwell3d->regs;
    const MTLScissorRect scissor = ScissorState(regs, framebuffer.Width(), framebuffer.Height());
    if (scissor.width == 0 || scissor.height == 0) {
        return false;
    }
    [encoder setViewport:ViewportState(regs)];
    [encoder setScissorRect:scissor];

    if (regs.gl_cull_test_enabled) {
        if (regs.gl_cull_face == Maxwell::CullFace::FrontAndBack) {
            // Metal can't cull both faces; nothing would be drawn anyway.
            return false;
        }
        [encoder setCullMode:MaxwellToMTL::CullFace(regs.gl_cull_face)];
    } else {
        [encoder setCullMode:MTLCullModeNone];
    }
    // Same as the Vulkan renderer: a flipped window origin flips the winding.
    MTLWinding winding = MaxwellToMTL::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        winding = winding == MTLWindingClockwise ? MTLWindingCounterClockwise
                                                 : MTLWindingClockwise;
    }
    [encoder setFrontFacingWinding:winding];
    [encoder setTriangleFillMode:regs.polygon_mode_front == Maxwell::PolygonMode::Line
                                     ? MTLTriangleFillModeLines
                                     : MTLTriangleFillModeFill];

    const bool depth_bias = regs.polygon_offset_fill_enable || regs.polygon_offset_line_enable ||
                            regs.polygon_offset_point_enable;
    [encoder setDepthBias:depth_bias ? regs.depth_bias / 2.0f : 0.0f
               slopeScale:depth_bias ? regs.slope_scale_depth_bias : 0.0f
                    clamp:depth_bias ? regs.depth_bias_clamp : 0.0f];
    [encoder setBlendColorRed:regs.blend_color.r
                        green:regs.blend_color.g
                         blue:regs.blend_color.b
                        alpha:regs.blend_color.a];

    [encoder setDepthStencilState:DepthStencilState(framebuffer)];
    const bool two_sided = regs.stencil_two_side_enable != 0;
    [encoder setStencilFrontReferenceValue:regs.stencil_front_ref & 0xFF
                        backReferenceValue:(two_sided ? regs.stencil_back_ref
                                                      : regs.stencil_front_ref) &
                                           0xFF];
    return true;
}

id<MTLDepthStencilState> RasterizerMetal::DepthStencilState(const Framebuffer& framebuffer) {
    const auto& regs = maxwell3d->regs;
    const bool has_depth = framebuffer.HasAspectDepthBit();
    const bool has_stencil = framebuffer.HasAspectStencilBit();
    const bool depth_test = has_depth && regs.depth_test_enable != 0;
    const bool depth_write = has_depth && regs.depth_write_enabled != 0;
    const bool stencil = has_stencil && regs.stencil_enable != 0;
    const bool two_sided = regs.stencil_two_side_enable != 0;
    const auto& back_op = two_sided ? regs.stencil_back_op : regs.stencil_front_op;
    const u32 back_func_mask =
        two_sided ? regs.stencil_back_func_mask : regs.stencil_front_func_mask;
    const u32 back_mask = two_sided ? regs.stencil_back_mask : regs.stencil_front_mask;

    const std::array<u32, 13> key{
        depth_test ? 1U : 0U,
        depth_write ? 1U : 0U,
        depth_test ? static_cast<u32>(regs.depth_test_func) : 0U,
        stencil ? 1U : 0U,
        stencil ? static_cast<u32>(regs.stencil_front_op.fail) : 0U,
        stencil ? static_cast<u32>(regs.stencil_front_op.zfail) : 0U,
        stencil ? static_cast<u32>(regs.stencil_front_op.zpass) : 0U,
        stencil ? static_cast<u32>(regs.stencil_front_op.func) : 0U,
        stencil ? (regs.stencil_front_func_mask & 0xFF) | ((regs.stencil_front_mask & 0xFF) << 8)
                : 0U,
        stencil ? static_cast<u32>(back_op.fail) : 0U,
        stencil ? static_cast<u32>(back_op.zfail) : 0U,
        stencil ? static_cast<u32>(back_op.zpass) | (static_cast<u32>(back_op.func) << 16) : 0U,
        stencil ? (back_func_mask & 0xFF) | ((back_mask & 0xFF) << 8) : 0U,
    };
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(key.data()), sizeof(key));
    if (const auto it = depth_stencil_states.find(hash); it != depth_stencil_states.end()) {
        return it->second;
    }
    MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
    desc.depthCompareFunction =
        depth_test ? MaxwellToMTL::ComparisonOp(regs.depth_test_func) : MTLCompareFunctionAlways;
    desc.depthWriteEnabled = depth_write ? YES : NO;
    if (stencil) {
        desc.frontFaceStencil =
            StencilFace(regs.stencil_front_op.fail, regs.stencil_front_op.zfail,
                        regs.stencil_front_op.zpass, regs.stencil_front_op.func,
                        regs.stencil_front_func_mask, regs.stencil_front_mask);
        desc.backFaceStencil = StencilFace(back_op.fail, back_op.zfail, back_op.zpass,
                                           back_op.func, back_func_mask, back_mask);
    }
    id<MTLDepthStencilState> state = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
    depth_stencil_states.emplace(hash, state);
    return state;
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
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        pipeline_cache.InvalidateRegion(addr, size);
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
    for (const auto& [addr, size] : sequences) {
        pipeline_cache.InvalidateRegion(addr, size);
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
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
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
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
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
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
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
