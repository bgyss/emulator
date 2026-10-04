// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <TargetConditionals.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bit_cast.h"
#include "common/cityhash.h"
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
#include "video_core/surface.h"

namespace Metal {

namespace {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;

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
                             TextureCache& texture_cache_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      buffer_cache{buffer_cache_}, buffer_cache_runtime{buffer_cache_runtime_},
      texture_cache{texture_cache_} {
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

PipelineCache::~PipelineCache() = default;

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    std::ranges::transform(maxwell3d->regs.vertex_streams, graphics_key.vertex_strides.begin(),
                           [](const auto& stream) { return static_cast<u16>(stream.stride); });
    const auto [it, is_new] = graphics_cache.try_emplace(graphics_key);
    if (is_new) {
        it->second = CreateGraphicsPipeline();
    }
    return it->second.get();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    const GraphicsPipelineCacheKey& key = graphics_key;
    // Program indices: VertexA, VertexB, TessellationControl, TessellationEval, Geometry,
    // Fragment.
    if (key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0 || key.unique_hashes[4] != 0) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "Tessellation and geometry shaders are not supported on "
                                      "Metal; skipping their draws");
        }
        return nullptr;
    }
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, key.unique_hashes);
    const auto envs = environments.Span();
    main_pools.ReleaseContents();

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
            Shader::Maxwell::Flow::CFG cfg(env, main_pools.flow_block, cfg_offset, index == 0);
            if (!uses_vertex_a || index != 1) {
                programs[index] =
                    TranslateProgram(main_pools.inst, main_pools.block, env, cfg, host_info);
            } else {
                auto program_vb =
                    TranslateProgram(main_pools.inst, main_pools.block, env, cfg, host_info);
                programs[index] = MergeDualVertexPrograms(programs[0], program_vb, env);
            }
            if (programs[index].info.requires_layer_emulation) {
                throw std::runtime_error("layer output needs a geometry shader");
            }
        }

        std::array<std::optional<GraphicsStage>, GraphicsPipeline::NUM_STAGES> stages;
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
            const std::vector<u32> code = EmitSPIRV(profile, runtime_info, program, bindings);

            MslTranslationOptions options{
                .ios = TARGET_OS_IOS != 0,
                .first_buffer_index = 0,
                .flip_vertex_y = is_vertex,
                .texture_1d_as_2d = true,
            };
            if (is_vertex) {
                // Keep the indices of the vertex buffers this pipeline reads free.
                for (size_t attr = 0; attr < Maxwell::NumVertexAttributes; ++attr) {
                    const auto& attribute = key.state.attributes[attr];
                    if (attribute.enabled != 0 && program.info.loads.Generic(attr)) {
                        options.buffer_index_limit =
                            std::min(options.buffer_index_limit,
                                     VertexBufferIndex(attribute.buffer.Value()));
                    }
                }
            }
            MslTranslationResult result = TranslateSpirvToMsl(code, options);
            if (!result.translation) {
                throw std::runtime_error("SPIR-V to MSL translation failed: " + result.error);
            }
            id<MTLFunction> function = CompileFunction(*result.translation);
            stages[stage_index] = GraphicsStage{
                .function = function,
                .translation = std::move(*result.translation),
                .first_binding = first_binding,
            };
            previous_stage = &program;
        }
        if (!stages[GraphicsPipeline::VERTEX_STAGE]) {
            throw std::runtime_error("pipeline has no vertex shader");
        }
        return std::make_unique<GraphicsPipeline>(device, scheduler, buffer_cache,
                                                  buffer_cache_runtime, texture_cache, key,
                                                  std::move(stages), infos);
    } catch (const Shader::Exception& exception) {
        LOG_ERROR(Render_Metal, "Failed to build a pipeline: {}", exception.what());
    } catch (const std::exception& exception) {
        LOG_ERROR(Render_Metal, "Failed to build a pipeline: {}", exception.what());
    }
    return nullptr;
}

id<MTLFunction> PipelineCache::CompileFunction(const MslTranslation& translation) {
    const u64 hash = Common::CityHash64(translation.source.data(), translation.source.size());
    if (const auto it = functions.find(hash); it != functions.end()) {
        return it->second;
    }
    NSError* error = nil;
    id<MTLLibrary> library =
        [device.GetDevice() newLibraryWithSource:@(translation.source.c_str())
                                         options:nil
                                           error:&error];
    if (library == nil) {
        LOG_DEBUG(Render_Metal, "MSL that failed to compile:\n{}", translation.source);
        throw std::runtime_error(std::string{"MSL compilation failed: "} +
                                 error.localizedDescription.UTF8String);
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@(translation.entry_point.c_str())];
    if (function == nil) {
        throw std::runtime_error("MSL entry point " + translation.entry_point + " not found");
    }
    functions.emplace(hash, function);
    return function;
}

} // namespace Metal
