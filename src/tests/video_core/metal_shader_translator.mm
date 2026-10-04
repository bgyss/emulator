// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <cstdint>
#include <span>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "video_core/host_shaders/block_linear_unswizzle_2d_comp_spv.h"
#include "video_core/host_shaders/vulkan_present_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_vert_spv.h"
#include "video_core/host_shaders/vulkan_uint8_comp_spv.h"
#include "video_core/renderer_metal/mtl_shader_translator.h"

using namespace Metal;

namespace {

MslTranslation Translate(std::span<const u32> spirv, const MslTranslationOptions& options = {}) {
    MslTranslationResult result = TranslateSpirvToMsl(spirv, options);
    INFO(result.error);
    REQUIRE(result.translation.has_value());
    return std::move(*result.translation);
}

/// Compiles the MSL with the system Metal compiler and checks the entry point exists.
void RequireCompiles(const MslTranslation& translation) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        WARN("No Metal device; skipping MSL compilation");
        return;
    }
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:@(translation.source.c_str()) options:nil error:&error];
    INFO(translation.source);
    INFO((error != nil ? std::string{error.localizedDescription.UTF8String} : std::string{}));
    REQUIRE(library != nil);
    REQUIRE([library newFunctionWithName:@(translation.entry_point.c_str())] != nil);
}

} // Anonymous namespace

TEST_CASE("MSL translation maps a combined image sampler", "[video_core][metal]") {
    const MslTranslation translation = Translate(VULKAN_PRESENT_FRAG_SPV);
    REQUIRE(translation.entry_point == "main0");
    REQUIRE(translation.bindings.size() == 1);
    REQUIRE(translation.bindings[0].kind == MslResourceKind::CombinedImageSampler);
    REQUIRE(translation.bindings[0].index == 0);
    REQUIRE(translation.bindings[0].sampler_index == 0);
    REQUIRE_FALSE(translation.push_constant_buffer.has_value());
    RequireCompiles(translation);
}

TEST_CASE("MSL translation places push constants after reserved buffers", "[video_core][metal]") {
    const MslTranslation translation =
        Translate(VULKAN_PRESENT_VERT_SPV, {.first_buffer_index = 8});
    REQUIRE(translation.push_constant_buffer == 8u);
    RequireCompiles(translation);
}

TEST_CASE("MSL translation provides buffer sizes for runtime arrays", "[video_core][metal]") {
    const MslTranslation translation = Translate(VULKAN_UINT8_COMP_SPV);
    REQUIRE(translation.bindings.size() == 2);
    REQUIRE(translation.bindings[0].index == 0);
    REQUIRE(translation.bindings[1].index == 1);
    REQUIRE(translation.buffer_size_buffer == 2u);
    RequireCompiles(translation);
}

TEST_CASE("MSL translation aliases variables that share a binding", "[video_core][metal]") {
    // Binding 1 is declared five times with different element types.
    const MslTranslation translation = Translate(BLOCK_LINEAR_UNSWIZZLE_2D_COMP_SPV);
    REQUIRE(translation.bindings.size() == 3);
    REQUIRE(translation.bindings[1].binding == 1);
    REQUIRE(translation.bindings[1].kind == MslResourceKind::Buffer);
    RequireCompiles(translation);
}

TEST_CASE("MSL translation reports errors instead of throwing", "[video_core][metal]") {
    const u32 garbage[] = {0xdeadbeef, 1, 2, 3, 4, 5};
    const MslTranslationResult invalid = TranslateSpirvToMsl(garbage, {});
    REQUIRE_FALSE(invalid.translation.has_value());
    REQUIRE_FALSE(invalid.error.empty());

    const MslTranslationResult overflow =
        TranslateSpirvToMsl(VULKAN_UINT8_COMP_SPV, {.first_buffer_index = MAX_BUFFER_INDEX - 1});
    REQUIRE_FALSE(overflow.translation.has_value());
    REQUIRE_FALSE(overflow.error.empty());
}

TEST_CASE("MSL translation flips vertex Y on request", "[video_core][metal]") {
    const MslTranslation plain = Translate(VULKAN_PRESENT_VERT_SPV);
    REQUIRE(plain.source.find("gl_Position.y = -(") == std::string::npos);
    const MslTranslation flipped = Translate(VULKAN_PRESENT_VERT_SPV, {.flip_vertex_y = true});
    INFO(flipped.source);
    REQUIRE(flipped.source.find("gl_Position.y = -(") != std::string::npos);
    RequireCompiles(flipped);
}

TEST_CASE("MSL translation keeps buffers below the index limit", "[video_core][metal]") {
    // Two storage buffers plus the buffer size buffer.
    const MslTranslationResult too_few =
        TranslateSpirvToMsl(VULKAN_UINT8_COMP_SPV, {.buffer_index_limit = 2});
    REQUIRE_FALSE(too_few.translation.has_value());
    const MslTranslation fits = Translate(VULKAN_UINT8_COMP_SPV, {.buffer_index_limit = 3});
    REQUIRE(fits.buffer_size_buffer == 2u);
    RequireCompiles(fits);
}
