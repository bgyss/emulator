// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <tuple>

#include <fmt/format.h>
#include <spirv_msl.hpp>

#include "video_core/renderer_metal/mtl_shader_translator.h"

namespace Metal {

namespace {

struct ResourceRef {
    MslResourceKind kind;
    u32 set;
    u32 binding;
};

/// Assigns MSL indices to resources and registers them with the compiler.
class BindingAllocator {
public:
    BindingAllocator(spirv_cross::CompilerMSL& compiler_, u32 first_buffer_index)
        : compiler{compiler_}, model{compiler.get_execution_model()},
          next_buffer{first_buffer_index} {}

    void Assign(const ResourceRef& ref, MslTranslation& out) {
        spirv_cross::MSLResourceBinding msl{};
        msl.stage = model;
        msl.desc_set = ref.set;
        msl.binding = ref.binding;

        MslBinding binding{.set = ref.set, .binding = ref.binding, .kind = ref.kind};
        switch (ref.kind) {
        case MslResourceKind::Buffer:
            binding.index = AllocateBuffer();
            msl.msl_buffer = binding.index;
            break;
        case MslResourceKind::Texture:
            binding.index = Allocate(next_texture, MAX_TEXTURE_INDEX, "texture");
            msl.msl_texture = binding.index;
            break;
        case MslResourceKind::CombinedImageSampler:
            binding.index = Allocate(next_texture, MAX_TEXTURE_INDEX, "texture");
            binding.sampler_index = Allocate(next_sampler, MAX_SAMPLER_INDEX, "sampler");
            msl.msl_texture = binding.index;
            msl.msl_sampler = binding.sampler_index;
            break;
        case MslResourceKind::Sampler:
            binding.index = Allocate(next_sampler, MAX_SAMPLER_INDEX, "sampler");
            msl.msl_sampler = binding.index;
            break;
        }
        compiler.add_msl_resource_binding(msl);
        out.bindings.push_back(binding);
    }

    void AssignPushConstants(MslTranslation& out) {
        spirv_cross::MSLResourceBinding msl{};
        msl.stage = model;
        msl.desc_set = spirv_cross::kPushConstDescSet;
        msl.binding = spirv_cross::kPushConstBinding;
        msl.msl_buffer = AllocateBuffer();
        compiler.add_msl_resource_binding(msl);
        out.push_constant_buffer = msl.msl_buffer;
    }

    [[nodiscard]] u32 AllocateBuffer() {
        return Allocate(next_buffer, MAX_BUFFER_INDEX, "buffer");
    }

private:
    static u32 Allocate(u32& next, u32 limit, const char* what) {
        if (next >= limit) {
            throw spirv_cross::CompilerError(
                fmt::format("shader needs more than {} Metal {} slots", limit, what));
        }
        return next++;
    }

    spirv_cross::CompilerMSL& compiler;
    spv::ExecutionModel model;
    u32 next_buffer;
    u32 next_texture{};
    u32 next_sampler{};
};

void CollectResources(const spirv_cross::CompilerMSL& compiler,
                      const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
                      MslResourceKind kind, std::vector<ResourceRef>& out) {
    for (const spirv_cross::Resource& resource : resources) {
        out.push_back({
            .kind = kind,
            .set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet),
            .binding = compiler.get_decoration(resource.id, spv::DecorationBinding),
        });
    }
}

MslTranslation Translate(std::span<const u32> spirv, const MslTranslationOptions& options) {
    spirv_cross::CompilerMSL compiler(spirv.data(), spirv.size());

    auto msl_options = compiler.get_msl_options();
    msl_options.platform = options.ios ? spirv_cross::CompilerMSL::Options::iOS
                                       : spirv_cross::CompilerMSL::Options::macOS;
    msl_options.set_msl_version(3, 0);
    compiler.set_msl_options(msl_options);

    const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
    std::vector<ResourceRef> refs;
    CollectResources(compiler, resources.uniform_buffers, MslResourceKind::Buffer, refs);
    CollectResources(compiler, resources.storage_buffers, MslResourceKind::Buffer, refs);
    CollectResources(compiler, resources.sampled_images, MslResourceKind::CombinedImageSampler,
                     refs);
    CollectResources(compiler, resources.separate_images, MslResourceKind::Texture, refs);
    CollectResources(compiler, resources.storage_images, MslResourceKind::Texture, refs);
    CollectResources(compiler, resources.separate_samplers, MslResourceKind::Sampler, refs);
    // Several variables can share a binding: the SPIR-V backend declares one storage buffer
    // variable per element type over the same guest buffer. They alias one Metal argument.
    std::ranges::sort(refs, [](const ResourceRef& lhs, const ResourceRef& rhs) {
        return std::tie(lhs.set, lhs.binding, lhs.kind) < std::tie(rhs.set, rhs.binding, rhs.kind);
    });
    const auto [first_duplicate, last] = std::ranges::unique(refs, [](const ResourceRef& lhs,
                                                                      const ResourceRef& rhs) {
        return std::tie(lhs.set, lhs.binding, lhs.kind) == std::tie(rhs.set, rhs.binding, rhs.kind);
    });
    refs.erase(first_duplicate, last);

    MslTranslation out;
    BindingAllocator allocator(compiler, options.first_buffer_index);
    for (const ResourceRef& ref : refs) {
        allocator.Assign(ref, out);
    }
    if (!resources.push_constant_buffers.empty()) {
        allocator.AssignPushConstants(out);
    }

    // Whether the shader needs the buffer size buffer is only known after compiling, so reserve
    // the next buffer index for it and drop it again if it went unused.
    const u32 buffer_size_index = allocator.AllocateBuffer();
    msl_options.buffer_size_buffer_index = buffer_size_index;
    compiler.set_msl_options(msl_options);

    out.source = compiler.compile();
    const auto entry_points = compiler.get_entry_points_and_stages();
    if (!entry_points.empty()) {
        out.entry_point = compiler.get_cleansed_entry_point_name(
            entry_points.front().name, entry_points.front().execution_model);
    }
    if (compiler.needs_buffer_size_buffer()) {
        out.buffer_size_buffer = buffer_size_index;
    }
    return out;
}

} // Anonymous namespace

MslTranslationResult TranslateSpirvToMsl(std::span<const u32> spirv,
                                         const MslTranslationOptions& options) {
    try {
        return {.translation = Translate(spirv, options), .error = {}};
    } catch (const spirv_cross::CompilerError& error) {
        return {.translation = std::nullopt, .error = error.what()};
    }
}

} // namespace Metal
