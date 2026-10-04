// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_graphics_pipeline.h holds Objective-C types; include it from .mm files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_shader_translator.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

namespace Tegra {
class MemoryManager;
}

namespace Metal {

class Device;
class Scheduler;

/// Pipeline cache key: the shaders and the Maxwell state Metal bakes into a pipeline.
///
/// The fixed state is the Vulkan renderer's FixedPipelineState, which only packs Maxwell
/// registers. It is refreshed with every extended dynamic state feature enabled except blending
/// and vertex input, because Metal sets everything else (depth, stencil, culling, ...) on the
/// encoder; vertex strides, which those features would drop, are part of Metal's vertex
/// descriptor and kept here.
struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    Vulkan::FixedPipelineState state;
    std::array<u16, Tegra::Engines::Maxwell3D::Regs::NumVertexArrays> vertex_strides;

    [[nodiscard]] size_t Hash() const noexcept;
    [[nodiscard]] bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;
};

/// Vertex buffers sit at the top of the vertex stage's buffer table, below the last index:
/// guest vertex buffer b at index LAST_VERTEX_BUFFER_INDEX - b.
constexpr u32 LAST_VERTEX_BUFFER_INDEX = MAX_BUFFER_INDEX - 1;

[[nodiscard]] constexpr u32 VertexBufferIndex(u32 binding) {
    return LAST_VERTEX_BUFFER_INDEX - binding;
}

/// One compiled shader stage.
struct GraphicsStage {
    id<MTLFunction> function;
    MslTranslation translation;
    /// SPIR-V binding of the stage's first descriptor; its descriptors take consecutive
    /// bindings in the order uniform buffers, storage buffers, texture buffers, image buffers,
    /// textures, images.
    u32 first_binding{};
};

class GraphicsPipeline {
public:
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;
    static constexpr size_t VERTEX_STAGE = 0;
    static constexpr size_t FRAGMENT_STAGE = NUM_STAGES - 1;

    explicit GraphicsPipeline(const Device& device, Scheduler& scheduler, BufferCache& buffer_cache,
                              BufferCacheRuntime& buffer_cache_runtime, TextureCache& texture_cache,
                              const GraphicsPipelineCacheKey& key,
                              std::array<std::optional<GraphicsStage>, NUM_STAGES> stages,
                              const std::array<const Shader::Info*, NUM_STAGES>& infos);

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

    /// Prepares the caches for a draw with this pipeline, opens the render pass and binds the
    /// pipeline state and every resource. Call with the buffer and texture cache mutexes held.
    /// @returns The render encoder to draw with, or nil when the draw must be skipped.
    [[nodiscard]] id<MTLRenderCommandEncoder> Configure(Tegra::Engines::Maxwell3D& maxwell3d,
                                                        Tegra::MemoryManager& gpu_memory,
                                                        bool is_indexed);

    [[nodiscard]] const GraphicsPipelineCacheKey& Key() const noexcept {
        return key;
    }

private:
    struct StageBindings {
        /// Bindings of the stage's MSL resources, indexed by SPIR-V binding - first_binding.
        std::vector<std::optional<MslBinding>> by_binding;
    };

    /// Pipeline state for the attachments of the current framebuffer, created on first use.
    id<MTLRenderPipelineState> PipelineState(const Framebuffer& framebuffer);

    void BindStage(id<MTLRenderCommandEncoder> encoder, size_t stage,
                   std::span<const BufferDescriptor> buffers,
                   std::span<const VideoCommon::ImageViewInOut> views,
                   std::span<const VideoCommon::SamplerId> samplers);

    const Device& device;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    BufferCacheRuntime& buffer_cache_runtime;
    TextureCache& texture_cache;
    GraphicsPipelineCacheKey key;

    std::array<std::optional<GraphicsStage>, NUM_STAGES> stages;
    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<StageBindings, NUM_STAGES> stage_bindings;
    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};

    MTLVertexDescriptor* vertex_descriptor;
    /// Guest vertex buffers the vertex descriptor reads.
    u32 used_vertex_buffers{};
    std::unordered_map<u64, id<MTLRenderPipelineState>> pipeline_states;
    bool failed{};
};

} // namespace Metal
