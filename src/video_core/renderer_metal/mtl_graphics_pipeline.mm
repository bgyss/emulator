// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

#include <boost/container/small_vector.hpp>

#include "common/cityhash.h"
#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/maxwell_to_mtl.h"
#include "video_core/renderer_metal/mtl_clear_helper.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Metal {

namespace {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Tegra::Texture::TexturePair;
using VideoCommon::ImageViewInOut;
using VideoCommon::SamplerId;

/// Number of SPIR-V bindings a stage's descriptors take; see GraphicsStage::first_binding.
u32 NumBindings(const Shader::Info& info) {
    u32 count = static_cast<u32>(info.constant_buffer_descriptors.size());
    for (const auto& desc : info.storage_buffers_descriptors) {
        count += desc.count;
    }
    count += static_cast<u32>(info.texture_buffer_descriptors.size());
    count += static_cast<u32>(info.image_buffer_descriptors.size());
    count += static_cast<u32>(info.texture_descriptors.size());
    count += static_cast<u32>(info.image_descriptors.size());
    return count;
}

bool IsIntegerFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
        return true;
    default:
        return false;
    }
}

MTLColorWriteMask WriteMask(const Vulkan::FixedPipelineState::BlendingAttachment& attachment) {
    MTLColorWriteMask mask = MTLColorWriteMaskNone;
    if (attachment.mask_r) {
        mask |= MTLColorWriteMaskRed;
    }
    if (attachment.mask_g) {
        mask |= MTLColorWriteMaskGreen;
    }
    if (attachment.mask_b) {
        mask |= MTLColorWriteMaskBlue;
    }
    if (attachment.mask_a) {
        mask |= MTLColorWriteMaskAlpha;
    }
    return mask;
}

void LogOnce(bool& logged, const char* message) {
    if (!logged) {
        logged = true;
        LOG_WARNING(Render_Metal, "{}", message);
    }
}

} // Anonymous namespace

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    u64 hash = Common::CityHash64(reinterpret_cast<const char*>(unique_hashes.data()),
                                  sizeof(unique_hashes));
    hash ^= Common::CityHash64(reinterpret_cast<const char*>(&state), state.Size());
    hash ^= Common::CityHash64(reinterpret_cast<const char*>(vertex_strides.data()),
                               sizeof(vertex_strides)) *
            31;
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return unique_hashes == rhs.unique_hashes && state.Size() == rhs.state.Size() &&
           std::memcmp(&state, &rhs.state, state.Size()) == 0 &&
           vertex_strides == rhs.vertex_strides;
}

GraphicsPipeline::GraphicsPipeline(const Device& device_, Scheduler& scheduler_,
                                   BufferCache& buffer_cache_,
                                   BufferCacheRuntime& buffer_cache_runtime_,
                                   TextureCache& texture_cache_,
                                   const GraphicsPipelineCacheKey& key_,
                                   std::array<std::optional<GraphicsStage>, NUM_STAGES> stages_,
                                   const std::array<const Shader::Info*, NUM_STAGES>& infos)
    : device{device_}, scheduler{scheduler_}, buffer_cache{buffer_cache_},
      buffer_cache_runtime{buffer_cache_runtime_}, texture_cache{texture_cache_}, key{key_},
      stages{std::move(stages_)} {
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info* const info = infos[stage];
        if (!info || !stages[stage]) {
            continue;
        }
        stage_infos[stage] = *info;
        enabled_uniform_buffer_masks[stage] = info->constant_buffer_mask;
        std::ranges::copy(info->constant_buffer_used_sizes, uniform_buffer_sizes[stage].begin());

        const GraphicsStage& data = *stages[stage];
        auto& by_binding = stage_bindings[stage].by_binding;
        by_binding.resize(NumBindings(*info));
        for (const MslBinding& binding : data.translation.bindings) {
            if (binding.binding >= data.first_binding &&
                binding.binding - data.first_binding < by_binding.size()) {
                by_binding[binding.binding - data.first_binding] = binding;
            }
        }
    }

    // Vertex input: every attribute the vertex shader reads.
    vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    const Shader::Info& vertex_info = stage_infos[VERTEX_STAGE];
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& attribute = key.state.attributes[index];
        if (attribute.enabled == 0 || !vertex_info.loads.Generic(index)) {
            continue;
        }
        if (index >= 31) {
            throw std::runtime_error("vertex attribute 31 exceeds Metal's attribute limit");
        }
        const MTLVertexFormat format =
            MaxwellToMTL::VertexFormat(attribute.Type(), attribute.Size());
        if (format == MTLVertexFormatInvalid) {
            throw std::runtime_error("vertex attribute format has no Metal equivalent");
        }
        const u32 buffer = attribute.buffer;
        MTLVertexAttributeDescriptor* desc = vertex_descriptor.attributes[index];
        desc.format = format;
        desc.offset = attribute.offset;
        desc.bufferIndex = VertexBufferIndex(buffer);
        used_vertex_buffers |= 1U << buffer;
    }
    for (u32 buffer = 0; buffer < Maxwell::NumVertexArrays; ++buffer) {
        if ((used_vertex_buffers & (1U << buffer)) == 0) {
            continue;
        }
        MTLVertexBufferLayoutDescriptor* layout =
            vertex_descriptor.layouts[VertexBufferIndex(buffer)];
        const u32 stride = key.vertex_strides[buffer];
        const u32 divisor = key.state.binding_divisors[buffer];
        if (stride == 0) {
            // Every vertex reads the same element.
            layout.stepFunction = MTLVertexStepFunctionConstant;
            layout.stepRate = 0;
            layout.stride = 4;
        } else {
            if (stride % 4 != 0) {
                throw std::runtime_error("vertex stride is not a multiple of 4 bytes");
            }
            layout.stride = stride;
            if (divisor != 0) {
                layout.stepFunction = MTLVertexStepFunctionPerInstance;
                layout.stepRate = divisor;
            } else {
                layout.stepFunction = MTLVertexStepFunctionPerVertex;
            }
        }
    }
}

id<MTLRenderPipelineState> GraphicsPipeline::PipelineState(const Framebuffer& framebuffer) {
    const MTLPixelFormat depth_format = framebuffer.DepthFormat();
    u64 signature = Common::CityHash64(reinterpret_cast<const char*>(&depth_format),
                                       sizeof(depth_format));
    std::array<MTLPixelFormat, NUM_RT> color_formats{};
    for (size_t index = 0; index < NUM_RT; ++index) {
        color_formats[index] = framebuffer.ColorAttachment(index) != nil
                                   ? framebuffer.ColorFormat(index)
                                   : MTLPixelFormatInvalid;
    }
    signature ^= Common::CityHash64(reinterpret_cast<const char*>(color_formats.data()),
                                    sizeof(color_formats)) *
                 31;
    signature ^= framebuffer.Samples() << 56;
    if (const auto it = pipeline_states.find(signature); it != pipeline_states.end()) {
        return it->second;
    }

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = stages[VERTEX_STAGE]->function;
    if (stages[FRAGMENT_STAGE]) {
        desc.fragmentFunction = stages[FRAGMENT_STAGE]->function;
    }
    desc.vertexDescriptor = vertex_descriptor;
    desc.rasterSampleCount = framebuffer.Samples();
    desc.alphaToCoverageEnabled = key.state.alpha_to_coverage_enabled != 0;
    desc.alphaToOneEnabled = key.state.alpha_to_one_enabled != 0;
    for (size_t index = 0; index < NUM_RT; ++index) {
        MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[index];
        color.pixelFormat = color_formats[index];
        if (color_formats[index] == MTLPixelFormatInvalid) {
            continue;
        }
        const auto& blend = key.state.attachments[index];
        color.writeMask = WriteMask(blend);
        // Metal can't blend integer formats.
        color.blendingEnabled = blend.enable != 0 && !IsIntegerFormat(color_formats[index]);
        if (color.blendingEnabled) {
            color.rgbBlendOperation = MaxwellToMTL::BlendEquation(blend.EquationRGB());
            color.alphaBlendOperation = MaxwellToMTL::BlendEquation(blend.EquationAlpha());
            color.sourceRGBBlendFactor = MaxwellToMTL::BlendFactor(blend.SourceRGBFactor());
            color.destinationRGBBlendFactor = MaxwellToMTL::BlendFactor(blend.DestRGBFactor());
            color.sourceAlphaBlendFactor = MaxwellToMTL::BlendFactor(blend.SourceAlphaFactor());
            color.destinationAlphaBlendFactor = MaxwellToMTL::BlendFactor(blend.DestAlphaFactor());
        }
    }
    if (ClearHelper::HasDepth(depth_format)) {
        desc.depthAttachmentPixelFormat = depth_format;
    }
    if (ClearHelper::HasStencil(depth_format)) {
        desc.stencilAttachmentPixelFormat = depth_format;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> state =
        [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (state == nil) {
        LOG_ERROR(Render_Metal, "Failed to create a render pipeline: {}",
                  error.localizedDescription.UTF8String);
    }
    // Remember failures too, so they are logged once.
    pipeline_states.emplace(signature, state);
    return state;
}

id<MTLRenderCommandEncoder> GraphicsPipeline::Configure(Tegra::Engines::Maxwell3D& maxwell3d,
                                                        Tegra::MemoryManager& gpu_memory,
                                                        bool is_indexed) {
    if (failed) {
        return nil;
    }
    thread_local boost::container::small_vector<ImageViewInOut, 64> views;
    thread_local boost::container::small_vector<SamplerId, 64> samplers;
    views.clear();
    samplers.clear();

    texture_cache.SynchronizeGraphicsDescriptors();
    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);

    const auto& regs = maxwell3d.regs;
    const bool via_header_index = regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
    std::array<size_t, NUM_STAGES + 1> view_offsets{};
    std::array<size_t, NUM_STAGES + 1> sampler_offsets{};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        view_offsets[stage] = views.size();
        sampler_offsets[stage] = samplers.size();
        if (!stages[stage]) {
            continue;
        }
        const Shader::Info& info = stage_infos[stage];
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        size_t ssbo_index = 0;
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written);
            ++ssbo_index;
        }
        const auto& cbufs = maxwell3d.state.shader_stages[stage].const_buffers;
        const auto read_handle = [&](const auto& desc, u32 index) {
            const u32 index_offset = index << desc.size_shift;
            const GPUVAddr addr = cbufs[desc.cbuf_index].address + desc.cbuf_offset + index_offset;
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                          std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
                if (desc.has_secondary) {
                    const GPUVAddr separate_addr = cbufs[desc.secondary_cbuf_index].address +
                                                   desc.secondary_cbuf_offset + index_offset;
                    const u32 lhs_raw = gpu_memory.Read<u32>(addr) << desc.shift_left;
                    const u32 rhs_raw = gpu_memory.Read<u32>(separate_addr)
                                        << desc.secondary_shift_left;
                    return TexturePair(lhs_raw | rhs_raw, via_header_index);
                }
            }
            return TexturePair(gpu_memory.Read<u32>(addr), via_header_index);
        };
        const auto add_views = [&](const auto& desc, bool blacklist) {
            for (u32 index = 0; index < desc.count; ++index) {
                views.push_back({.index = read_handle(desc, index).first, .blacklist = blacklist});
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add_views(desc, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add_views(desc, false);
        }
        for (const auto& desc : info.texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto [image, sampler] = read_handle(desc, index);
                views.push_back({.index = image});
                samplers.push_back(image == 0 ? VideoCommon::NULL_SAMPLER_ID
                                              : texture_cache.GetGraphicsSamplerId(sampler));
            }
        }
        for (const auto& desc : info.image_descriptors) {
            add_views(desc, desc.is_written);
        }
    }
    view_offsets[NUM_STAGES] = views.size();
    sampler_offsets[NUM_STAGES] = samplers.size();
    texture_cache.FillGraphicsImageViews<true>(std::span(views.data(), views.size()));

    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!stages[stage]) {
            continue;
        }
        buffer_cache.UnbindGraphicsTextureBuffers(stage);
        const Shader::Info& info = stage_infos[stage];
        const ImageViewInOut* view = views.data() + view_offsets[stage];
        size_t index = 0;
        const auto add_buffer = [&](const auto& desc, bool is_image) {
            for (u32 i = 0; i < desc.count; ++i) {
                bool is_written = false;
                if constexpr (std::is_same_v<decltype(desc),
                                             const Shader::ImageBufferDescriptor&>) {
                    is_written = desc.is_written;
                }
                ImageView& image_view = texture_cache.GetImageView((view++)->id);
                buffer_cache.BindGraphicsTextureBuffer(stage, index, image_view.GpuAddr(),
                                                       image_view.BufferSize(), image_view.format,
                                                       is_written, is_image);
                ++index;
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add_buffer(desc, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add_buffer(desc, true);
        }
    }

    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);

    // The buffer cache records each stage's buffers in descriptor order.
    buffer_cache_runtime.ClearDescriptors();
    std::array<std::pair<size_t, size_t>, NUM_STAGES> buffer_ranges{};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!stages[stage]) {
            continue;
        }
        const size_t begin = buffer_cache_runtime.GetDescriptors().size();
        buffer_cache.BindHostStageBuffers(stage);
        buffer_ranges[stage] = {begin, buffer_cache_runtime.GetDescriptors().size()};
    }

    texture_cache.UpdateRenderTargets(false);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    id<MTLRenderPipelineState> state = PipelineState(*framebuffer);
    if (state == nil) {
        return nil;
    }
    // Everything that copies or uploads is done; the render pass starts here.
    id<MTLRenderCommandEncoder> encoder = scheduler.RenderEncoder(*framebuffer);
    [encoder setRenderPipelineState:state];

    const auto vertex_buffers = buffer_cache_runtime.GetVertexBuffers();
    for (u32 buffer = 0; buffer < vertex_buffers.size(); ++buffer) {
        if ((used_vertex_buffers & (1U << buffer)) == 0) {
            continue;
        }
        const VertexBufferBinding& binding = vertex_buffers[buffer];
        if (binding.buffer == nil) {
            // Never bound by the guest; read zeros instead of an unbound slot.
            [encoder setVertexBuffer:buffer_cache_runtime.NullBuffer()
                              offset:0
                             atIndex:VertexBufferIndex(buffer)];
            continue;
        }
        [encoder setVertexBuffer:binding.buffer
                          offset:binding.offset
                         atIndex:VertexBufferIndex(buffer)];
    }

    const auto descriptors = buffer_cache_runtime.GetDescriptors();
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!stages[stage]) {
            continue;
        }
        const auto [begin, end] = buffer_ranges[stage];
        BindStage(encoder, stage, descriptors.subspan(begin, end - begin),
                  std::span(views.data() + view_offsets[stage],
                            view_offsets[stage + 1] - view_offsets[stage]),
                  std::span(samplers.data() + sampler_offsets[stage],
                            sampler_offsets[stage + 1] - sampler_offsets[stage]));
    }
    return encoder;
}

void GraphicsPipeline::BindStage(id<MTLRenderCommandEncoder> encoder, size_t stage,
                                 std::span<const BufferDescriptor> buffers,
                                 std::span<const ImageViewInOut> views,
                                 std::span<const SamplerId> samplers) {
    const GraphicsStage& data = *stages[stage];
    const Shader::Info& info = stage_infos[stage];
    const auto& by_binding = stage_bindings[stage].by_binding;
    const bool is_vertex = stage == VERTEX_STAGE;

    const auto set_buffer = [&](id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
        if (is_vertex) {
            [encoder setVertexBuffer:buffer offset:offset atIndex:index];
        } else {
            [encoder setFragmentBuffer:buffer offset:offset atIndex:index];
        }
    };
    const auto set_bytes = [&](const void* bytes, NSUInteger length, NSUInteger index) {
        if (is_vertex) {
            [encoder setVertexBytes:bytes length:length atIndex:index];
        } else {
            [encoder setFragmentBytes:bytes length:length atIndex:index];
        }
    };
    const auto set_texture = [&](id<MTLTexture> texture, NSUInteger index) {
        if (is_vertex) {
            [encoder setVertexTexture:texture atIndex:index];
        } else {
            [encoder setFragmentTexture:texture atIndex:index];
        }
    };
    const auto set_sampler = [&](id<MTLSamplerState> sampler, NSUInteger index) {
        if (is_vertex) {
            [encoder setVertexSamplerState:sampler atIndex:index];
        } else {
            [encoder setFragmentSamplerState:sampler atIndex:index];
        }
    };

    // Sizes of the bound buffers, for shaders that read runtime array lengths.
    std::array<u32, MAX_BUFFER_INDEX> buffer_sizes{};
    size_t binding = 0;
    size_t buffer_index = 0;
    const auto bind_buffers = [&](u32 count) {
        const std::optional<MslBinding>& msl =
            binding < by_binding.size() ? by_binding[binding] : std::nullopt;
        for (u32 i = 0; i < count && buffer_index < buffers.size(); ++i, ++buffer_index) {
            if (!msl) {
                continue;
            }
            const BufferDescriptor& buffer = buffers[buffer_index];
            const u32 index = msl->index + i;
            set_buffer(buffer.buffer, buffer.offset, index);
            if (index < buffer_sizes.size()) {
                buffer_sizes[index] = buffer.size;
            }
        }
        ++binding;
    };
    for (size_t i = 0; i < info.constant_buffer_descriptors.size(); ++i) {
        bind_buffers(1);
    }
    for (const auto& desc : info.storage_buffers_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            bind_buffers(1);
        }
    }
    // Texture and image buffers need texture views of their buffers, which aren't implemented.
    static bool logged_texture_buffers{};
    for (const auto& desc : info.texture_buffer_descriptors) {
        LogOnce(logged_texture_buffers, "Texture buffers are not implemented on Metal yet");
        buffer_index += desc.count;
        ++binding;
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        LogOnce(logged_texture_buffers, "Texture buffers are not implemented on Metal yet");
        buffer_index += desc.count;
        ++binding;
    }

    size_t view_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                        Shader::NumDescriptors(info.image_buffer_descriptors);
    size_t sampler_index = 0;
    const ImageView& null_view = texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID);
    for (const auto& desc : info.texture_descriptors) {
        const std::optional<MslBinding>& msl =
            binding < by_binding.size() ? by_binding[binding] : std::nullopt;
        for (u32 i = 0; i < desc.count; ++i) {
            ImageView& image_view = texture_cache.GetImageView(views[view_index++].id);
            const Sampler& sampler = texture_cache.GetSampler(samplers[sampler_index++]);
            if (!msl) {
                continue;
            }
            id<MTLTexture> texture = image_view.Handle(desc.type);
            if (texture == nil) {
                texture = null_view.Handle(desc.type);
            }
            set_texture(texture, msl->index + i);
            const bool use_fallback = sampler.HasAddedAnisotropy() &&
                                      !image_view.SupportsAnisotropy();
            set_sampler(use_fallback ? sampler.HandleWithDefaultAnisotropy() : sampler.Handle(),
                        msl->sampler_index + i);
        }
        ++binding;
    }
    for (const auto& desc : info.image_descriptors) {
        const std::optional<MslBinding>& msl =
            binding < by_binding.size() ? by_binding[binding] : std::nullopt;
        for (u32 i = 0; i < desc.count; ++i) {
            ImageView& image_view = texture_cache.GetImageView(views[view_index++].id);
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            if (msl) {
                set_texture(image_view.Handle(desc.type), msl->index + i);
            }
        }
        ++binding;
    }

    if (data.translation.push_constant_buffer) {
        // No resolution scaling: no texture or image is rescaled, and the down factor is 1.
        Shader::Backend::SPIRV::RescalingLayout rescaling{};
        rescaling.down_factor = std::bit_cast<u32>(1.0f);
        set_bytes(&rescaling, sizeof(rescaling), *data.translation.push_constant_buffer);
    }
    if (data.translation.buffer_size_buffer) {
        set_bytes(buffer_sizes.data(), sizeof(buffer_sizes), *data.translation.buffer_size_buffer);
    }
}

} // namespace Metal
