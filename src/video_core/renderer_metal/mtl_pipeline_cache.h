// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_pipeline_cache.h holds Objective-C types; include it from .mm files only"
#endif

#import <Metal/Metal.h>

#include <memory>
#include <unordered_map>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"
#include "video_core/shader_cache.h"

namespace Metal {

class Device;
class Scheduler;
struct MslTranslation;

/**
 * Builds Metal pipelines from guest shaders: the shader recompiler emits SPIR-V, which
 * SPIRV-Cross translates to MSL, which Metal compiles at runtime. Compiled functions are shared
 * by every pipeline whose MSL is identical.
 *
 * Only vertex and fragment shaders are supported; pipelines with tessellation or geometry
 * shaders fail and their draws are skipped.
 */
class PipelineCache final : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, const Device& device,
                           Scheduler& scheduler, BufferCache& buffer_cache,
                           BufferCacheRuntime& buffer_cache_runtime, TextureCache& texture_cache);
    ~PipelineCache();

    /// The pipeline for the current Maxwell state, or nullptr when it can't be built.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

private:
    struct KeyHash {
        size_t operator()(const GraphicsPipelineCacheKey& key) const noexcept {
            return key.Hash();
        }
    };

    struct ShaderPools {
        void ReleaseContents() {
            flow_block.ReleaseContents();
            block.ReleaseContents();
            inst.ReleaseContents();
        }

        Shader::ObjectPool<Shader::IR::Inst> inst{8192};
        Shader::ObjectPool<Shader::IR::Block> block{32};
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
    };

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// Compiles MSL into a function, reusing an earlier function with the same source.
    id<MTLFunction> CompileFunction(const MslTranslation& translation);

    const Device& device;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    BufferCacheRuntime& buffer_cache_runtime;
    TextureCache& texture_cache;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    Vulkan::DynamicFeatures dynamic_features{};
    ShaderPools main_pools;

    GraphicsPipelineCacheKey graphics_key{};
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>, KeyHash>
        graphics_cache;
    std::unordered_map<u64, id<MTLFunction>> functions;
};

} // namespace Metal
