// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Metal {

/// Metal's per-stage argument table limits (without argument buffers).
constexpr u32 MAX_BUFFER_INDEX = 31;
constexpr u32 MAX_TEXTURE_INDEX = 128;
constexpr u32 MAX_SAMPLER_INDEX = 16;

enum class MslResourceKind {
    /// A uniform or storage buffer, bound with set*Buffer.
    Buffer,
    /// A sampled, storage or texel-buffer image, bound with set*Texture.
    Texture,
    /// A combined image sampler: bound with set*Texture at index and set*SamplerState at
    /// sampler_index.
    CombinedImageSampler,
    /// A standalone sampler, bound with set*SamplerState.
    Sampler,
};

/// Where a SPIR-V (descriptor set, binding) ended up in the MSL argument tables.
struct MslBinding {
    u32 set{};
    u32 binding{};
    MslResourceKind kind{};
    u32 index{};
    /// Only meaningful for CombinedImageSampler.
    u32 sampler_index{};
    /// Array length; an array takes this many consecutive indices from index (and from
    /// sampler_index).
    u32 count{1};
};

struct MslTranslationOptions {
    /// Target iOS instead of macOS.
    bool ios{};
    /// First buffer index the shader's resources may use. Vertex shaders share the buffer table
    /// with vertex buffers, so the caller can reserve the low indices for those.
    u32 first_buffer_index{};
    /// One past the last buffer index the shader's resources may use, to keep the high indices
    /// free for vertex buffers.
    u32 buffer_index_limit{MAX_BUFFER_INDEX};
    /// Negate clip-space Y in vertex outputs. The shader recompiler targets Vulkan, whose clip
    /// space has Y pointing down; Metal's points up.
    bool flip_vertex_y{};
    /// Declare 1D images as 2D textures one texel high, the way the texture cache stores them.
    bool texture_1d_as_2d{};
};

struct MslTranslation {
    std::string source;
    /// Entry point name in the MSL source. SPIRV-Cross renames "main", which is reserved in MSL.
    std::string entry_point;
    std::vector<MslBinding> bindings;
    /// Buffer index of the push constant block, if the shader has one.
    std::optional<u32> push_constant_buffer;
    /// Buffer index of the array of buffer sizes the shader reads to implement the length of
    /// runtime arrays, if it needs one. Each entry is the size in bytes of the buffer at that
    /// index, as a uint.
    std::optional<u32> buffer_size_buffer;
};

struct MslTranslationResult {
    std::optional<MslTranslation> translation;
    /// Why translation failed, when translation is empty.
    std::string error;
};

/**
 * Translates a SPIR-V module to Metal Shading Language with SPIRV-Cross.
 *
 * Resources are assigned MSL argument table indices in (set, binding) order: buffers from
 * options.first_buffer_index, then the push constant block and the buffer size buffer after
 * them; textures and samplers from 0. Translation fails when a table overflows or SPIRV-Cross
 * rejects the module (for example geometry shaders, which Metal does not have).
 */
[[nodiscard]] MslTranslationResult TranslateSpirvToMsl(std::span<const u32> spirv,
                                                       const MslTranslationOptions& options);

} // namespace Metal
