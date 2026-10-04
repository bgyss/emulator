// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_pipeline_cache.h holds Objective-C types; include it from .mm files only"
#endif

#import <Metal/Metal.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"
#include "video_core/shader_cache.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Metal {

class Device;
class Scheduler;

/**
 * Builds Metal pipelines from guest shaders: the shader recompiler emits SPIR-V, which
 * SPIRV-Cross translates to MSL, which Metal compiles at runtime. Compiled functions are shared
 * by every pipeline whose MSL is identical.
 *
 * With asynchronous shader building on, pipelines compile on worker threads and large draws
 * skip them until they are ready. With the disk shader cache on, the guest shaders of every
 * pipeline and the framebuffer formats it was used with are saved, and LoadDiskResources builds
 * them all at boot.
 *
 * Only vertex and fragment shaders are supported; pipelines with tessellation or geometry
 * shaders fail and their draws are skipped.
 */
class PipelineCache final : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, const Device& device,
                           Scheduler& scheduler, BufferCache& buffer_cache,
                           BufferCacheRuntime& buffer_cache_runtime, TextureCache& texture_cache,
                           VideoCore::ShaderNotify& shader_notify);
    ~PipelineCache();

    /// The pipeline for the current Maxwell state, or nullptr when it can't be built.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    /// Builds the pipelines saved for this title, and saves new ones from now on.
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// Whether a draw may skip a pipeline that is still compiling instead of waiting for it.
    [[nodiscard]] bool MaySkipDraw(u32 vertex_or_index_count) const noexcept;

    /// Logs the pipeline summary every so many pipelines.
    void TickFrame();

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

    /// Builds the pipeline for the current Maxwell state and saves it to disk.
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// Runs the shader recompiler on the stages and creates the pipeline, which compiles the
    /// result to Metal now or in the background.
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, bool build_in_background);

    /// Appends a pipeline state's formats to the state cache file.
    void SaveState(u64 key_hash, const AttachmentFormats& formats);

    /// Reads the state cache file: the framebuffer formats each pipeline was used with.
    std::unordered_map<u64, std::vector<AttachmentFormats>> LoadStates();

    void LogStatistics(const char* when);

    const Device& device;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    BufferCacheRuntime& buffer_cache_runtime;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    const bool use_asynchronous_shaders;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    Vulkan::DynamicFeatures dynamic_features{};
    ShaderPools main_pools;

    GraphicsPipelineCacheKey graphics_key{};
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>, KeyHash>
        graphics_cache;

    PipelineCompiler compiler;
    u64 logged_pipeline_count{};

    std::filesystem::path pipeline_cache_filename;
    std::filesystem::path state_cache_filename;
    std::mutex state_file_mutex;

    // Declared last so they are destroyed first, while what their tasks use still exists.
    Common::ThreadWorker serialization_thread;
    Common::ThreadWorker workers;
};

} // namespace Metal
