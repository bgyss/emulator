// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_graphics_pipeline.h holds Objective-C types; include it from .mm files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_shader_translator.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

namespace Tegra {
class MemoryManager;
}

namespace VideoCore {
class ShaderNotify;
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

/// A stage's SPIR-V, waiting to be translated to MSL and compiled.
struct StageSource {
    std::vector<u32> spirv;
    MslTranslationOptions options;
    /// SPIR-V binding of the stage's first descriptor; see GraphicsStage::first_binding.
    u32 first_binding{};
};

/// One compiled shader stage.
struct GraphicsStage {
    id<MTLFunction> function;
    MslTranslation translation;
    /// SPIR-V binding of the stage's first descriptor; its descriptors take consecutive
    /// bindings in the order uniform buffers, storage buffers, texture buffers, image buffers,
    /// textures, images.
    u32 first_binding{};
};

/// Pixel formats and sample count of a framebuffer's attachments, which Metal bakes into
/// pipeline states. Plain integers, so it can be written to the pipeline state cache on disk.
struct AttachmentFormats {
    std::array<u32, NUM_RT> color{};
    u32 depth{};
    u32 samples{1};

    [[nodiscard]] static AttachmentFormats From(const Framebuffer& framebuffer);

    [[nodiscard]] u64 Hash() const noexcept;

    [[nodiscard]] bool operator==(const AttachmentFormats&) const noexcept = default;
};

/// Counters for the pipeline summary in the log. Times are in nanoseconds.
struct PipelineStatistics {
    std::atomic<u64> pipelines_built{};
    std::atomic<u64> pipelines_loaded{};
    std::atomic<u64> pipelines_failed{};
    std::atomic<u64> functions_compiled{};
    std::atomic<u64> functions_shared{};
    std::atomic<u64> states_built{};
    std::atomic<u64> draws_skipped{};
    std::atomic<u64> recompile_ns{};
    std::atomic<u64> translate_ns{};
    std::atomic<u64> compile_ns{};
    std::atomic<u64> state_ns{};
    /// Time the GPU thread spent waiting for pipelines and pipeline states.
    std::atomic<u64> wait_ns{};
    /// Longest the GPU thread waited for one pipeline or pipeline state.
    std::atomic<u64> max_wait_ns{};

    void AddWait(u64 ns) noexcept;
};

/**
 * Turns stage SPIR-V into Metal functions, sharing functions whose MSL is identical, and holds
 * what pipelines need to build in the background. Thread-safe.
 */
class PipelineCompiler {
public:
    using StateBuiltCallback = std::function<void(u64 key_hash, const AttachmentFormats&)>;

    explicit PipelineCompiler(const Device& device);
    ~PipelineCompiler();

    /// Translates a stage to MSL and compiles it.
    /// @throws std::runtime_error when translation or compilation fails.
    [[nodiscard]] GraphicsStage Compile(const StageSource& source);

    [[nodiscard]] const Device& GetDevice() const noexcept {
        return device;
    }

    [[nodiscard]] PipelineStatistics& Statistics() noexcept {
        return statistics;
    }

    /// Threads that build pipelines in the background, or nullptr to build them on the GPU
    /// thread.
    Common::ThreadWorker* workers{};
    /// Counts pipelines built in the background for the status bar; may be nullptr.
    VideoCore::ShaderNotify* shader_notify{};
    /// Called on whichever thread built a new pipeline state, to remember it on disk.
    StateBuiltCallback on_state_built;

private:
    const Device& device;
    std::mutex mutex;
    std::unordered_map<u64, id<MTLFunction>> functions;
    PipelineStatistics statistics;
};

/**
 * A graphics pipeline: Metal functions, a vertex descriptor and binding tables built from the
 * pipeline's shader stages, plus one Metal pipeline state per framebuffer format signature.
 *
 * The functions can be compiled on a worker thread. Until they are, draws either skip the
 * pipeline or wait for it; a waiting draw compiles the pipeline itself when no worker has
 * started on it. Pipeline states are created the same way.
 */
class GraphicsPipeline {
public:
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;
    static constexpr size_t VERTEX_STAGE = 0;
    static constexpr size_t FRAGMENT_STAGE = NUM_STAGES - 1;

    /// @param build_in_background Queue the compilation on the compiler's workers instead of
    ///                            compiling before returning.
    explicit GraphicsPipeline(const Device& device, Scheduler& scheduler, BufferCache& buffer_cache,
                              BufferCacheRuntime& buffer_cache_runtime, TextureCache& texture_cache,
                              PipelineCompiler& compiler, const GraphicsPipelineCacheKey& key,
                              std::array<std::optional<StageSource>, NUM_STAGES> sources,
                              const std::array<const Shader::Info*, NUM_STAGES>& infos,
                              bool build_in_background);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

    /// Prepares the caches for a draw with this pipeline, opens the render pass and binds the
    /// pipeline state and every resource. Call with the buffer and texture cache mutexes held.
    /// @param may_skip Skip the draw instead of waiting when the pipeline or its pipeline state
    ///                 for the current framebuffer isn't built yet; their builds are queued.
    /// @returns The render encoder to draw with, or nil when the draw must be skipped.
    [[nodiscard]] id<MTLRenderCommandEncoder> Configure(Tegra::Engines::Maxwell3D& maxwell3d,
                                                        Tegra::MemoryManager& gpu_memory,
                                                        bool is_indexed, bool may_skip);

    /// Creates the pipeline states for these framebuffer formats on the calling thread, unless
    /// they exist. For pipelines loaded from disk.
    void PrebuildStates(std::span<const AttachmentFormats> formats);

    [[nodiscard]] bool IsBuilt() const noexcept {
        return built.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFailed() const noexcept {
        return failed.load(std::memory_order_acquire);
    }

    [[nodiscard]] const GraphicsPipelineCacheKey& Key() const noexcept {
        return key;
    }

private:
    struct StageBindings {
        /// Bindings of the stage's MSL resources, indexed by SPIR-V binding - first_binding.
        std::vector<std::optional<MslBinding>> by_binding;
    };

    /// Compiles the stages and builds the vertex descriptor and binding tables, then the
    /// pipeline states draws asked for meanwhile. Only the first call does anything.
    void Build();

    /// Waits until the pipeline is built, building it on this thread if no one has started.
    void WaitBuilt();

    void BuildStages();

    /// Creates the pipeline state for these formats and publishes it.
    void BuildState(const AttachmentFormats& formats, bool remember);

    /// The pipeline state for these formats, or nil when it failed or is still being built and
    /// may_skip is set.
    id<MTLRenderPipelineState> PipelineState(const AttachmentFormats& formats, bool may_skip);

    /// Asks for the pipeline state for these formats to be built once the pipeline is.
    void RequestState(const AttachmentFormats& formats);

    void BindStage(id<MTLRenderCommandEncoder> encoder, size_t stage,
                   std::span<const BufferDescriptor> buffers,
                   std::span<const VideoCommon::ImageViewInOut> views,
                   std::span<const VideoCommon::SamplerId> samplers);

    const Device& device;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    BufferCacheRuntime& buffer_cache_runtime;
    TextureCache& texture_cache;
    PipelineCompiler& compiler;
    GraphicsPipelineCacheKey key;
    u64 key_hash{};

    std::array<std::optional<StageSource>, NUM_STAGES> sources;
    std::array<std::optional<GraphicsStage>, NUM_STAGES> stages;
    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<bool, NUM_STAGES> stage_enabled{};
    std::array<StageBindings, NUM_STAGES> stage_bindings;
    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};

    MTLVertexDescriptor* vertex_descriptor;
    /// Guest vertex buffers the vertex descriptor reads.
    u32 used_vertex_buffers{};

    /// Set by whoever builds the pipeline, so it's built once.
    std::atomic<bool> build_claimed{};
    std::atomic<bool> built{};
    std::atomic<bool> failed{};
    bool notify_shader_built{};

    /// Guards everything below and signals builds finishing.
    std::mutex state_mutex;
    std::condition_variable state_condvar;
    std::unordered_map<u64, id<MTLRenderPipelineState>> pipeline_states;
    /// Pipeline states being built, by AttachmentFormats::Hash.
    std::unordered_set<u64> pending_states;
    /// Pipeline states asked for before the pipeline was built.
    std::vector<AttachmentFormats> requested_states;
};

} // namespace Metal
