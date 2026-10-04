// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/mtl_clear_helper.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

using namespace Metal;
using Kind = ClearHelper::ColorKind;

namespace {

constexpr NSUInteger SIZE = 4;
constexpr MTLScissorRect WHOLE{0, 0, 1U << 20, 1U << 20};

std::unique_ptr<Device> MakeDevice() {
    try {
        return std::make_unique<Device>();
    } catch (const std::runtime_error& error) {
        WARN("No Metal device; skipping: " << error.what());
        return nullptr;
    }
}

id<MTLTexture> MakeTarget(const Device& device, MTLPixelFormat format) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                   width:SIZE
                                                                                  height:SIZE
                                                                               mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return [device.GetDevice() newTextureWithDescriptor:desc];
}

/// Reads back a 4x4 texture of 4-byte texels.
std::vector<u32> ReadBack(const Device& device, Scheduler& scheduler, id<MTLTexture> texture) {
    id<MTLBuffer> buffer = [device.GetDevice() newBufferWithLength:SIZE * SIZE * 4
                                                           options:MTLResourceStorageModeShared];
    [scheduler.BlitEncoder() copyFromTexture:texture
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake(0, 0, 0)
                                  sourceSize:MTLSizeMake(SIZE, SIZE, 1)
                                    toBuffer:buffer
                           destinationOffset:0
                      destinationBytesPerRow:SIZE * 4
                    destinationBytesPerImage:0];
    scheduler.Finish();
    std::vector<u32> texels(SIZE * SIZE);
    std::memcpy(texels.data(), buffer.contents, texels.size() * sizeof(u32));
    return texels;
}

bool Inside(NSUInteger x, NSUInteger y, MTLScissorRect rect) {
    return x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height;
}

} // Anonymous namespace

TEST_CASE("Metal clears whole color attachments", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    ClearHelper helper(*device, scheduler);
    id<MTLTexture> target = MakeTarget(*device, MTLPixelFormatRGBA8Unorm);

    helper.ClearColor(target, 0, WHOLE, 0xF, Kind::Float, {1.0, 0.0, 0.0, 1.0});
    for (const u32 texel : ReadBack(*device, scheduler, target)) {
        REQUIRE(texel == 0xFF0000FFU);
    }
}

TEST_CASE("Metal clears scissored and masked color", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    ClearHelper helper(*device, scheduler);
    id<MTLTexture> target = MakeTarget(*device, MTLPixelFormatRGBA8Unorm);
    helper.ClearColor(target, 0, WHOLE, 0xF, Kind::Float, {0.0, 0.0, 0.0, 0.0});

    // Green inside the scissor, then white through a red+alpha mask everywhere.
    const MTLScissorRect rect{1, 1, 2, 2};
    helper.ClearColor(target, 0, rect, 0xF, Kind::Float, {0.0, 1.0, 0.0, 0.0});
    helper.ClearColor(target, 0, WHOLE, 0x1 | 0x8, Kind::Float, {1.0, 1.0, 1.0, 1.0});

    const std::vector<u32> texels = ReadBack(*device, scheduler, target);
    for (NSUInteger y = 0; y < SIZE; ++y) {
        for (NSUInteger x = 0; x < SIZE; ++x) {
            INFO("x " << x << " y " << y);
            const u32 green = Inside(x, y, rect) ? 0x0000FF00U : 0;
            REQUIRE(texels[y * SIZE + x] == (0xFF0000FFU | green));
        }
    }
}

TEST_CASE("Metal clears integer color attachments", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    ClearHelper helper(*device, scheduler);
    id<MTLTexture> target = MakeTarget(*device, MTLPixelFormatRGBA8Uint);
    helper.ClearColor(target, 0, WHOLE, 0xF, Kind::Uint, {0, 0, 0, 0});
    // A scissored clear takes the draw path with the integer fragment function.
    const MTLScissorRect rect{0, 0, 2, 4};
    helper.ClearColor(target, 0, rect, 0xF, Kind::Uint, {1, 2, 3, 4});

    const std::vector<u32> texels = ReadBack(*device, scheduler, target);
    for (NSUInteger y = 0; y < SIZE; ++y) {
        for (NSUInteger x = 0; x < SIZE; ++x) {
            INFO("x " << x << " y " << y);
            REQUIRE(texels[y * SIZE + x] == (Inside(x, y, rect) ? 0x04030201U : 0U));
        }
    }
}

TEST_CASE("Metal clears depth attachments", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    ClearHelper helper(*device, scheduler);
    id<MTLTexture> target = MakeTarget(*device, MTLPixelFormatDepth32Float);
    helper.ClearDepthStencil(target, 0, WHOLE, true, 0.25f, false, 0, 0);
    const MTLScissorRect rect{2, 0, 2, 4};
    helper.ClearDepthStencil(target, 0, rect, true, 0.75f, false, 0, 0);

    const std::vector<u32> texels = ReadBack(*device, scheduler, target);
    for (NSUInteger y = 0; y < SIZE; ++y) {
        for (NSUInteger x = 0; x < SIZE; ++x) {
            INFO("x " << x << " y " << y);
            float depth;
            std::memcpy(&depth, &texels[y * SIZE + x], sizeof(depth));
            REQUIRE(depth == (Inside(x, y, rect) ? 0.75f : 0.25f));
        }
    }
}

TEST_CASE("Metal clears depth-stencil attachments with a stencil mask", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Scheduler scheduler(*device);
    ClearHelper helper(*device, scheduler);
    id<MTLTexture> target = MakeTarget(*device, MTLPixelFormatDepth32Float_Stencil8);
    REQUIRE(ClearHelper::HasDepth(target.pixelFormat));
    REQUIRE(ClearHelper::HasStencil(target.pixelFormat));
    // Whole clear, then a masked stencil clear that has to draw.
    helper.ClearDepthStencil(target, 0, WHOLE, true, 1.0f, true, 0, 0xFF);
    helper.ClearDepthStencil(target, 0, WHOLE, false, 0.0f, true, 0x5A, 0x0F);
    scheduler.Finish();
    SUCCEED("Depth-stencil clear pipelines were created and ran");
}
