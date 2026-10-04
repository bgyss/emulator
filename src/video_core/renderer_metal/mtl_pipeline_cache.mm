// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <TargetConditionals.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/bit_cast.h"
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_shader_translator.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"

namespace Metal {

namespace {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;

using Clock = std::chrono::steady_clock;

/// Version of the transferable pipeline cache, metal.bin. Bump it when the key changes.
constexpr u32 PIPELINE_CACHE_VERSION = 1;
/// Version of the pipeline state cache, metal_states.bin. Bump it when the pixel format
/// mapping or the pipeline state descriptor changes.
constexpr u32 STATE_CACHE_VERSION = 1;
constexpr std::array<char, 8> STATE_CACHE_MAGIC{'c', 'i', 't', 'r', 'm', 't', 'l', 's'};

/// Draws with at most this many vertices or indices wait for their pipeline even with
/// asynchronous shaders: they are mostly full-screen passes and UI, where a skipped draw shows.
constexpr u32 MAX_WAITING_DRAW_COUNT = 32;

/// Log the pipeline summary every this many new pipelines.
constexpr u64 STATISTICS_INTERVAL = 100;

struct StateRecord {
    u64 key_hash;
    AttachmentFormats formats;
};
static_assert(std::has_unique_object_representations_v<StateRecord>);

u64 ElapsedNs(Clock::time_point start) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

double Milliseconds(u64 ns) {
    return static_cast<double>(ns) / 1'000'000.0;
}

size_t NumPipelineWorkers() {
    // Leave a core each for the GPU thread and the CPU emulation.
    const size_t threads = std::max<size_t>(std::thread::hardware_concurrency(), 2);
    return std::max<size_t>(threads > 4 ? threads - 2 : threads / 2, 1);
}

// The helpers below follow the Vulkan pipeline cache.

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    return Shader::CompareFunction::Always;
}

Shader::AttributeType CastAttributeType(const Vulkan::FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::FragmentOutputType GetFragmentOutputType(u8 encoded_format) {
    const auto format = static_cast<Tegra::RenderTargetFormat>(encoded_format);
    if (format == Tegra::RenderTargetFormat::NONE) {
        return Shader::FragmentOutputType::Float;
    }
    const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
    if (!VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
        return Shader::FragmentOutputType::Float;
    }
    return VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
               ? Shader::FragmentOutputType::SignedInt
               : Shader::FragmentOutputType::UnsignedInt;
}

Shader::RuntimeInfo MakeRuntimeInfo(const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
    } else {
        info.previous_stage_stores.mask.set();
    }
    const auto topology = key.state.topology.Value();
    switch (program.stage) {
    case Shader::Stage::VertexB:
        if (topology == Maxwell::PrimitiveTopology::Points) {
            info.fixed_state_point_size = Common::BitCast<float>(key.state.point_size);
        }
        info.convert_depth_mode = key.state.ndc_minus_one_to_one != 0;
        std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                               &CastAttributeType);
        break;
    case Shader::Stage::Fragment:
        std::ranges::transform(key.state.color_formats, info.frag_color_types.begin(),
                               &GetFragmentOutputType);
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = Common::BitCast<float>(key.state.alpha_test_ref);
        break;
    default:
        break;
    }
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    default:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             BufferCache& buffer_cache_, BufferCacheRuntime& buffer_cache_runtime_,
                             TextureCache& texture_cache_,
                             VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      buffer_cache{buffer_cache_}, buffer_cache_runtime{buffer_cache_runtime_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      compiler{device_}, serialization_thread(1, "MtlPipelineSerialization"),
      workers(NumPipelineWorkers(), "MtlPipelineBuilder") {
    compiler.workers = &workers;
    compiler.shader_notify = &shader_notify;
    compiler.on_state_built = [this](u64 key_hash, const AttachmentFormats& formats) {
        SaveState(key_hash, formats);
    };
    // What SPIR-V the shader recompiler may emit for SPIRV-Cross to translate to MSL.
    profile = Shader::Profile{
        .supported_spirv = 0x00010500,
        .unified_descriptor_binding = true,
        .has_split_descriptor_sets = false,
        .support_descriptor_aliasing = true,
        .support_int8 = true,
        .support_int16 = true,
        .support_int64 = true,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_vote = true,
        .support_viewport_index_layer_non_geometry = true,
        .support_viewport_mask = false,
        .support_typeless_image_loads = false,
        .support_demote_to_helper_invocation = false,
        .support_int64_atomics = false,
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = false,
        .support_native_ndc = false,
        .support_scaled_attributes = false,
        .support_multi_viewport = false,
        .support_geometry_streams = false,
        .warp_size_potentially_larger_than_guest = false,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_fastmath_off = false,
        .min_ssbo_alignment = 16,
        .max_user_clip_distances = 8,
    };
    host_info = Shader::HostTranslateInfo{
        // Metal has no 64-bit floats.
        .support_float64 = false,
        .support_float16 = true,
        .support_int64 = true,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = true,
        .min_ssbo_alignment = 16,
        .support_geometry_shader_passthrough = false,
        .support_conditional_barrier = false,
    };
    // Metal sets these on the render encoder, so they stay out of the pipeline key. Blending
    // and vertex input are pipeline state in Metal.
    dynamic_features = Vulkan::DynamicFeatures{
        .has_extended_dynamic_state = true,
        .has_extended_dynamic_state_2 = true,
        .has_extended_dynamic_state_2_extra = true,
        .has_extended_dynamic_state_3_blend = false,
        .has_extended_dynamic_state_3_enables = true,
        .has_dynamic_vertex_input = false,
        .has_transform_feedback = false,
    };
}

PipelineCache::~PipelineCache() {
    // Pipelines still compiling are dropped with the workers; finish writing the disk cache.
    serialization_thread.WaitForRequests();
    LogStatistics("at shutdown");
}

bool PipelineCache::MaySkipDraw(u32 vertex_or_index_count) const noexcept {
    return use_asynchronous_shaders && vertex_or_index_count > MAX_WAITING_DRAW_COUNT;
}

void PipelineCache::TickFrame() {
    const PipelineStatistics& stats = compiler.Statistics();
    const u64 count = stats.pipelines_built + stats.pipelines_failed;
    if (count >= logged_pipeline_count + STATISTICS_INTERVAL) {
        logged_pipeline_count = count;
        LogStatistics("so far");
    }
}

void PipelineCache::LogStatistics(const char* when) {
    const PipelineStatistics& stats = compiler.Statistics();
    const u64 built = stats.pipelines_built;
    if (built == 0 && stats.pipelines_failed == 0) {
        return;
    }
    const u64 functions = std::max<u64>(stats.functions_compiled, 1);
    const u64 states = std::max<u64>(stats.states_built, 1);
    const u64 pipelines = std::max<u64>(built + stats.pipelines_failed, 1);
    LOG_INFO(Render_Metal,
             "Pipelines {}: {} built ({} from the disk cache), {} failed; {} Metal functions "
             "compiled, {} shared; {} pipeline states. Average per pipeline: recompile {:.1f} ms; "
             "per function: MSL translation {:.1f} ms, Metal compile {:.1f} ms; per state: "
             "{:.1f} ms. The GPU thread waited {:.0f} ms in total (longest {:.0f} ms); {} draws "
             "skipped while compiling",
             when, built, stats.pipelines_loaded.load(), stats.pipelines_failed.load(),
             stats.functions_compiled.load(), stats.functions_shared.load(),
             stats.states_built.load(), Milliseconds(stats.recompile_ns) / pipelines,
             Milliseconds(stats.translate_ns) /
                 std::max<u64>(stats.functions_compiled + stats.functions_shared, 1),
             Milliseconds(stats.compile_ns) / functions, Milliseconds(stats.state_ns) / states,
             Milliseconds(stats.wait_ns), Milliseconds(stats.max_wait_ns),
             stats.draws_skipped.load());
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    std::ranges::transform(maxwell3d->regs.vertex_streams, graphics_key.vertex_strides.begin(),
                           [](const auto& stream) { return static_cast<u16>(stream.stride); });
    const auto [it, is_new] = graphics_cache.try_emplace(graphics_key);
    if (is_new) {
        const auto start = Clock::now();
        it->second = CreateGraphicsPipeline();
        compiler.Statistics().AddWait(ElapsedNs(start));
    }
    return it->second.get();
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::ShaderDir);
    const auto base_dir = shader_dir / fmt::format("{:016x}", title_id);
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename = base_dir / "metal.bin";
    state_cache_filename = base_dir / "metal_states.bin";
    const auto saved_states = LoadStates();

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        size_t skipped{};
    } state;
    const auto start = Clock::now();

    const auto load_compute = [&](std::ifstream&, FileEnvironment) {
        // Metal doesn't save compute pipelines.
        ++state.skipped;
    };
    const auto load_graphics = [&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!std::ranges::all_of(envs, &FileEnvironment::HasValidEntryInstruction)) {
            ++state.skipped;
            return;
        }
        std::vector<AttachmentFormats> formats;
        if (const auto it = saved_states.find(key.Hash()); it != saved_states.end()) {
            formats = it->second;
        }
        workers.QueueWork([this, key, envs_ = std::move(envs), formats_ = std::move(formats),
                           &state, &callback]() mutable {
            @autoreleasepool {
                ShaderPools pools;
                boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
                for (auto& env : envs_) {
                    env_ptrs.push_back(&env);
                }
                auto pipeline = CreateGraphicsPipeline(
                    pools, key, std::span(env_ptrs.data(), env_ptrs.size()), false);
                if (pipeline) {
                    pipeline->PrebuildStates(formats_);
                    if (!pipeline->IsFailed()) {
                        ++compiler.Statistics().pipelines_loaded;
                    }
                }
                std::scoped_lock lock{state.mutex};
                if (pipeline) {
                    graphics_cache.emplace(key, std::move(pipeline));
                }
                ++state.built;
                if (state.has_loaded) {
                    callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
                }
            }
        });
        ++state.total;
    };
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, PIPELINE_CACHE_VERSION,
                               load_compute, load_graphics);
    if (state.skipped != 0) {
        LOG_WARNING(Render_Metal, "Skipped {} invalid pipelines in the disk cache", state.skipped);
    }
    LOG_INFO(Render_Metal, "Building {} pipelines from the disk cache", state.total);
    {
        std::scoped_lock lock{state.mutex};
        graphics_cache.reserve(graphics_cache.size() + state.total);
        callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
        state.has_loaded = true;
    }
    workers.WaitForRequests(stop_loading);
    if (state.total != 0) {
        LOG_INFO(Render_Metal, "Built {} pipelines from the disk cache in {:.1f} s", state.total,
                 Milliseconds(ElapsedNs(start)) / 1000.0);
        LogStatistics("after loading the disk cache");
        logged_pipeline_count =
            compiler.Statistics().pipelines_built + compiler.Statistics().pipelines_failed;
    }
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    main_pools.ReleaseContents();
    auto pipeline = CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(),
                                           use_asynchronous_shaders);
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key = graphics_key,
                                    envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        VideoCommon::SerializePipeline(key, env_ptrs, pipeline_cache_filename,
                                       PIPELINE_CACHE_VERSION);
    });
    return pipeline;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, bool build_in_background) {
    // Program indices: VertexA, VertexB, TessellationControl, TessellationEval, Geometry,
    // Fragment.
    if (key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0 || key.unique_hashes[4] != 0) {
        static std::atomic<bool> logged{};
        if (!logged.exchange(true)) {
            LOG_WARNING(Render_Metal, "Tessellation and geometry shaders are not supported on "
                                      "Metal; skipping their draws");
        }
        return nullptr;
    }
    const auto start = Clock::now();
    try {
        std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
        const bool uses_vertex_a = key.unique_hashes[0] != 0;
        const bool uses_vertex_b = key.unique_hashes[1] != 0;
        size_t env_index = 0;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::Environment& env = *envs[env_index++];
            const u32 cfg_offset =
                static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            if (!uses_vertex_a || index != 1) {
                programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
            } else {
                auto program_vb = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
                programs[index] = MergeDualVertexPrograms(programs[0], program_vb, env);
            }
            if (programs[index].info.requires_layer_emulation) {
                throw std::runtime_error("layer output needs a geometry shader");
            }
        }

        std::array<std::optional<StageSource>, GraphicsPipeline::NUM_STAGES> sources;
        std::array<const Shader::Info*, GraphicsPipeline::NUM_STAGES> infos{};
        Shader::Backend::Bindings bindings;
        const Shader::IR::Program* previous_stage = nullptr;
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0 || index == 0) {
                continue;
            }
            Shader::IR::Program& program = programs[index];
            const size_t stage_index = index - 1;
            infos[stage_index] = &program.info;
            const bool is_vertex = program.stage == Shader::Stage::VertexB;

            const Shader::RuntimeInfo runtime_info = MakeRuntimeInfo(key, program, previous_stage);
            ConvertLegacyToGeneric(program, runtime_info);
            const u32 first_binding = bindings.unified;
            StageSource source{
                .spirv = EmitSPIRV(profile, runtime_info, program, bindings),
                .options =
                    MslTranslationOptions{
                        .ios = TARGET_OS_IOS != 0,
                        .first_buffer_index = 0,
                        .flip_vertex_y = is_vertex,
                        .texture_1d_as_2d = true,
                    },
                .first_binding = first_binding,
            };
            if (is_vertex) {
                // Keep the indices of the vertex buffers this pipeline reads free.
                for (size_t attr = 0; attr < Maxwell::NumVertexAttributes; ++attr) {
                    const auto& attribute = key.state.attributes[attr];
                    if (attribute.enabled != 0 && program.info.loads.Generic(attr)) {
                        source.options.buffer_index_limit =
                            std::min(source.options.buffer_index_limit,
                                     VertexBufferIndex(attribute.buffer.Value()));
                    }
                }
            }
            sources[stage_index] = std::move(source);
            previous_stage = &program;
        }
        if (!sources[GraphicsPipeline::VERTEX_STAGE]) {
            throw std::runtime_error("pipeline has no vertex shader");
        }
        compiler.Statistics().recompile_ns += ElapsedNs(start);
        // The pipeline copies the infos, so the programs can go when this returns.
        auto pipeline = std::make_unique<GraphicsPipeline>(
            device, scheduler, buffer_cache, buffer_cache_runtime, texture_cache, compiler, key,
            std::move(sources), infos, build_in_background);
        if (!build_in_background) {
            LOG_DEBUG(Render_Metal, "Built pipeline {:016x} in {:.1f} ms", key.Hash(),
                      Milliseconds(ElapsedNs(start)));
        }
        return pipeline;
    } catch (const Shader::Exception& exception) {
        LOG_ERROR(Render_Metal, "Failed to build pipeline {:016x}: {}", key.Hash(),
                  exception.what());
    } catch (const std::exception& exception) {
        LOG_ERROR(Render_Metal, "Failed to build pipeline {:016x}: {}", key.Hash(),
                  exception.what());
    }
    ++compiler.Statistics().pipelines_failed;
    return nullptr;
}

void PipelineCache::SaveState(u64 key_hash, const AttachmentFormats& formats) {
    if (state_cache_filename.empty()) {
        return;
    }
    serialization_thread.QueueWork([this, record = StateRecord{key_hash, formats}] {
        std::scoped_lock lock{state_file_mutex};
        std::ofstream file(state_cache_filename, std::ios::binary | std::ios::ate | std::ios::app);
        if (!file.is_open()) {
            return;
        }
        if (file.tellp() == 0) {
            file.write(STATE_CACHE_MAGIC.data(), STATE_CACHE_MAGIC.size())
                .write(reinterpret_cast<const char*>(&STATE_CACHE_VERSION),
                       sizeof(STATE_CACHE_VERSION));
        }
        file.write(reinterpret_cast<const char*>(&record), sizeof(record));
    });
}

std::unordered_map<u64, std::vector<AttachmentFormats>> PipelineCache::LoadStates() {
    std::unordered_map<u64, std::vector<AttachmentFormats>> states;
    std::scoped_lock lock{state_file_mutex};
    std::ifstream file(state_cache_filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return states;
    }
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::array<char, 8> magic{};
    u32 version{};
    file.read(magic.data(), magic.size()).read(reinterpret_cast<char*>(&version), sizeof(version));
    const size_t header_size = magic.size() + sizeof(version);
    if (!file || magic != STATE_CACHE_MAGIC || version != STATE_CACHE_VERSION ||
        (size - header_size) % sizeof(StateRecord) != 0) {
        file.close();
        LOG_INFO(Render_Metal, "Deleting an old or invalid pipeline state cache");
        Common::FS::RemoveFile(state_cache_filename);
        return states;
    }
    std::vector<StateRecord> records((size - header_size) / sizeof(StateRecord));
    file.read(reinterpret_cast<char*>(records.data()),
              static_cast<std::streamsize>(records.size() * sizeof(StateRecord)));
    for (const StateRecord& record : records) {
        auto& formats = states[record.key_hash];
        if (std::ranges::find(formats, record.formats) == formats.end()) {
            formats.push_back(record.formats);
        }
    }
    return states;
}

} // namespace Metal
