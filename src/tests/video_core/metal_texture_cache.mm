// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/maxwell_to_mtl.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/texture_cache/image_view_info.h"

using namespace Metal;
using Tegra::Texture::SwizzleSource;
using VideoCommon::BufferImageCopy;
using VideoCommon::ImageCopy;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
using VideoCommon::Region2D;
using VideoCommon::SubresourceRange;
using VideoCore::Surface::PixelFormat;

namespace {

std::unique_ptr<Device> MakeDevice() {
    try {
        return std::make_unique<Device>();
    } catch (const std::runtime_error& error) {
        WARN("No Metal device; skipping: " << error.what());
        return nullptr;
    }
}

ImageInfo MakeInfo(PixelFormat format, u32 width, u32 height, s32 levels = 1, s32 layers = 1) {
    ImageInfo info;
    info.format = format;
    info.type = ImageType::e2D;
    info.size = {width, height, 1};
    info.resources = {.levels = levels, .layers = layers};
    return info;
}

BufferImageCopy FullCopy(u32 width, u32 height, s32 level = 0, s32 layer = 0, s32 layers = 1,
                         size_t offset = 0) {
    return {
        .buffer_offset = offset,
        .buffer_size = size_t{width} * height * 4 * static_cast<size_t>(layers),
        .buffer_row_length = width,
        .buffer_image_height = height,
        .image_subresource = {.base_level = level, .base_layer = layer, .num_layers = layers},
        .image_offset = {0, 0, 0},
        .image_extent = {width, height, 1},
    };
}

struct Fixture {
    explicit Fixture(const Device& device_)
        : device{device_}, scheduler{device}, pool{device, scheduler},
          runtime{device, scheduler, pool} {}

    void Upload(Image& image, std::span<const u8> data, std::span<const BufferImageCopy> copies) {
        const StagingBufferRef staging = runtime.UploadStagingBuffer(data.size());
        std::memcpy(staging.mapped_span.data(), data.data(), data.size());
        image.UploadMemory(staging, copies);
    }

    std::vector<u8> Download(Image& image, size_t size, std::span<const BufferImageCopy> copies) {
        const StagingBufferRef staging = runtime.DownloadStagingBuffer(size);
        image.DownloadMemory(staging, copies);
        runtime.Finish();
        const u8* const data = staging.mapped_span.data();
        return std::vector<u8>(data, data + size);
    }

    const Device& device;
    Scheduler scheduler;
    StagingBufferPool pool;
    TextureCacheRuntime runtime;
};

std::vector<u8> Iota(size_t size, u8 first = 0) {
    std::vector<u8> data(size);
    std::iota(data.begin(), data.end(), first);
    return data;
}

std::vector<u32> Words(std::span<const u8> bytes) {
    std::vector<u32> words(bytes.size() / sizeof(u32));
    std::memcpy(words.data(), bytes.data(), words.size() * sizeof(u32));
    return words;
}

std::span<const u8> Bytes(std::span<const u32> words) {
    return {reinterpret_cast<const u8*>(words.data()), words.size_bytes()};
}

} // Anonymous namespace

TEST_CASE("Metal format table covers every guest format", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    for (size_t index = 0; index < VideoCore::Surface::MaxPixelFormat; ++index) {
        const auto format = static_cast<PixelFormat>(index);
        const MaxwellToMTL::FormatInfo info = MaxwellToMTL::SurfaceFormat(*device, format);
        INFO("format " << index);
        if (format == PixelFormat::R32G32B32_FLOAT || format == PixelFormat::G4R4_UNORM) {
            // No Metal texture format; the image falls back to RGBA8.
            REQUIRE(info.format == MTLPixelFormatInvalid);
        } else {
            REQUIRE(info.format != MTLPixelFormatInvalid);
        }
    }
}

TEST_CASE("Metal images and views can be created for every guest format", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    for (size_t index = 0; index < VideoCore::Surface::MaxPixelFormat; ++index) {
        const auto format = static_cast<PixelFormat>(index);
        INFO("format " << index);
        std::unique_ptr<Image> image;
        REQUIRE_NOTHROW(image = std::make_unique<Image>(fixture.runtime,
                                                        MakeInfo(format, 64, 64, 2, 1), 0, 0));
        REQUIRE(image->Handle() != nil);
        REQUIRE(image->Handle().mipmapLevelCount == 2);

        const ImageViewInfo view_info(ImageViewType::e2D, format,
                                      SubresourceRange{.base = {0, 0}, .extent = {2, 1}});
        const ImageView view(fixture.runtime, view_info, VideoCommon::ImageId{1}, *image);
        REQUIRE(view.Handle(Shader::TextureType::Color2D) != nil);
        REQUIRE(view.Handle(Shader::TextureType::ColorArray2D) != nil);
        REQUIRE(view.RenderTarget() != nil);
    }
}

TEST_CASE("Metal images round-trip levels and layers", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image image(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 16, 8, 2, 2), 0, 0);
    REQUIRE(image.Handle().textureType == MTLTextureType2DArray);
    REQUIRE(image.Handle().arrayLength == 2);

    // Level 0 of both layers, then level 1 (8x4) of both layers.
    const size_t level0 = 16 * 8 * 4 * 2;
    const size_t level1 = 8 * 4 * 4 * 2;
    const std::array copies{
        FullCopy(16, 8, 0, 0, 2, 0),
        FullCopy(8, 4, 1, 0, 2, level0),
    };
    const std::vector<u8> data = Iota(level0 + level1, 3);
    fixture.Upload(image, data, copies);
    REQUIRE(fixture.Download(image, data.size(), copies) == data);
}

TEST_CASE("Metal images copy between images of the same format", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image src(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 8, 8), 0, 0);
    Image dst(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 8, 8), 0, 0);
    const std::array full{FullCopy(8, 8)};

    std::vector<u32> pixels(64);
    std::iota(pixels.begin(), pixels.end(), 1U);
    fixture.Upload(src, Bytes(pixels), full);
    const std::vector<u32> zeros(64, 0);
    fixture.Upload(dst, Bytes(zeros), full);

    // The top-left 4x4 of src lands at (4, 4) in dst.
    const std::array copies{ImageCopy{
        .src_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .dst_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .src_offset = {0, 0, 0},
        .dst_offset = {4, 4, 0},
        .extent = {4, 4, 1},
    }};
    fixture.runtime.CopyImage(dst, src, copies);

    const std::vector<u32> result = Words(fixture.Download(dst, 256, full));
    for (u32 y = 0; y < 8; ++y) {
        for (u32 x = 0; x < 8; ++x) {
            INFO("x " << x << " y " << y);
            const u32 expected = (x >= 4 && y >= 4) ? pixels[(y - 4) * 8 + (x - 4)] : 0;
            REQUIRE(result[y * 8 + x] == expected);
        }
    }
}

TEST_CASE("Metal images copy bits between formats of the same size", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image src(fixture.runtime, MakeInfo(PixelFormat::R32_UINT, 4, 4), 0, 0);
    Image dst(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 4, 4), 0, 0);
    const std::array full{FullCopy(4, 4)};

    std::vector<u32> pixels(16);
    std::iota(pixels.begin(), pixels.end(), 0x01020300U);
    fixture.Upload(src, Bytes(pixels), full);

    const std::array copies{ImageCopy{
        .src_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .dst_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .src_offset = {0, 0, 0},
        .dst_offset = {0, 0, 0},
        .extent = {4, 4, 1},
    }};
    fixture.runtime.CopyImage(dst, src, copies);
    REQUIRE(Words(fixture.Download(dst, 64, full)) == pixels);
}

TEST_CASE("Metal images reinterpret depth as color", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image depth(fixture.runtime, MakeInfo(PixelFormat::D32_FLOAT, 4, 4), 0, 0);
    Image color(fixture.runtime, MakeInfo(PixelFormat::R32_FLOAT, 4, 4), 0, 0);
    const std::array full{FullCopy(4, 4)};

    std::array<float, 16> values;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) / 16.0f;
    }
    fixture.Upload(depth, std::span{reinterpret_cast<const u8*>(values.data()), sizeof(values)},
                   full);
    REQUIRE(fixture.runtime.ShouldReinterpret(color, depth));
    const std::array copies{ImageCopy{
        .src_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .dst_subresource = {.base_level = 0, .base_layer = 0, .num_layers = 1},
        .src_offset = {0, 0, 0},
        .dst_offset = {0, 0, 0},
        .extent = {4, 4, 1},
    }};
    fixture.runtime.ReinterpretImage(color, depth, copies);

    const std::vector<u8> result = fixture.Download(color, sizeof(values), full);
    REQUIRE(std::memcmp(result.data(), values.data(), sizeof(values)) == 0);
}

TEST_CASE("Metal blits scale and flip images", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image src(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 2, 2), 0, 0);
    Image dst(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 4, 4), 0, 0);
    const std::array pixels{0xFF0000FFU, 0xFF00FF00U, 0xFFFF0000U, 0xFFFFFFFFU};
    fixture.Upload(src, Bytes(pixels), std::array{FullCopy(2, 2)});

    const ImageViewInfo src_info(ImageViewType::e2D, PixelFormat::A8B8G8R8_UNORM);
    const ImageViewInfo dst_info(ImageViewType::e2D, PixelFormat::A8B8G8R8_UNORM);
    ImageView src_view(fixture.runtime, src_info, VideoCommon::ImageId{1}, src);
    ImageView dst_view(fixture.runtime, dst_info, VideoCommon::ImageId{2}, dst);
    using Fermi2D = Tegra::Engines::Fermi2D;

    // Nearest 2x upscale: each source texel covers a 2x2 block.
    fixture.runtime.BlitImage(nullptr, dst_view, src_view,
                              Region2D{.start = {0, 0}, .end = {4, 4}},
                              Region2D{.start = {0, 0}, .end = {2, 2}}, Fermi2D::Filter::Point,
                              Fermi2D::Operation::SrcCopy);
    std::vector<u32> result = Words(fixture.Download(dst, 64, std::array{FullCopy(4, 4)}));
    for (u32 y = 0; y < 4; ++y) {
        for (u32 x = 0; x < 4; ++x) {
            INFO("x " << x << " y " << y);
            REQUIRE(result[y * 4 + x] == pixels[(y / 2) * 2 + x / 2]);
        }
    }

    // The same blit with the destination rows reversed flips the image vertically.
    fixture.runtime.BlitImage(nullptr, dst_view, src_view,
                              Region2D{.start = {0, 4}, .end = {4, 0}},
                              Region2D{.start = {0, 0}, .end = {2, 2}}, Fermi2D::Filter::Point,
                              Fermi2D::Operation::SrcCopy);
    result = Words(fixture.Download(dst, 64, std::array{FullCopy(4, 4)}));
    for (u32 y = 0; y < 4; ++y) {
        for (u32 x = 0; x < 4; ++x) {
            INFO("x " << x << " y " << y);
            REQUIRE(result[y * 4 + x] == pixels[(1 - y / 2) * 2 + x / 2]);
        }
    }
}

TEST_CASE("Metal null image views read as zero", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    const ImageView view(fixture.runtime, VideoCommon::NullImageViewParams{});
    for (u32 index = 0; index < Shader::NUM_TEXTURE_TYPES; ++index) {
        INFO("texture type " << index);
        const auto type = static_cast<Shader::TextureType>(index);
        REQUIRE((view.Handle(type) != nil) != (type == Shader::TextureType::Buffer));
    }
    u32 texel = 0xDEADBEEF;
    [view.Handle(Shader::TextureType::Color2D) getBytes:&texel
                                            bytesPerRow:sizeof(texel)
                                          bytesPerImage:sizeof(texel)
                                             fromRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
                                            mipmapLevel:0
                                                  slice:0];
    REQUIRE(texel == 0);
}

TEST_CASE("Metal samplers are created from TSC entries", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Tegra::Texture::TSCEntry tsc{};
    tsc.wrap_u.Assign(Tegra::Texture::WrapMode::Wrap);
    tsc.wrap_v.Assign(Tegra::Texture::WrapMode::Border);
    tsc.wrap_p.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    tsc.depth_compare_enabled.Assign(1);
    tsc.depth_compare_func.Assign(Tegra::Texture::DepthCompareFunc::LessEqual);
    tsc.max_lod_clamp.Assign(4 * 256);

    const Metal::Sampler sampler(fixture.runtime, tsc);
    REQUIRE(sampler.Handle() != nil);
    REQUIRE(sampler.HandleWithDefaultAnisotropy() != nil);
}

TEST_CASE("Metal framebuffers describe their attachments", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Image color(fixture.runtime, MakeInfo(PixelFormat::A8B8G8R8_UNORM, 32, 16), 0, 0);
    Image depth(fixture.runtime, MakeInfo(PixelFormat::D24_UNORM_S8_UINT, 32, 16), 0, 0);
    ImageView color_view(fixture.runtime,
                         ImageViewInfo(ImageViewType::e2D, PixelFormat::A8B8G8R8_UNORM),
                         VideoCommon::ImageId{1}, color);
    ImageView depth_view(fixture.runtime,
                         ImageViewInfo(ImageViewType::e2D, PixelFormat::D24_UNORM_S8_UINT),
                         VideoCommon::ImageId{2}, depth);

    std::array<ImageView*, VideoCommon::NUM_RT> color_buffers{};
    color_buffers[1] = &color_view;
    VideoCommon::RenderTargets key{};
    key.size = {32, 16};
    const Metal::Framebuffer framebuffer(fixture.runtime, color_buffers, &depth_view, key);

    REQUIRE(framebuffer.Width() == 32);
    REQUIRE(framebuffer.Height() == 16);
    REQUIRE(framebuffer.HasAspectDepthBit());
    REQUIRE(framebuffer.HasAspectStencilBit());
    REQUIRE(framebuffer.DepthFormat() == MTLPixelFormatDepth32Float_Stencil8);
    MTLRenderPassDescriptor* desc = framebuffer.MakeRenderPassDescriptor();
    REQUIRE(desc.colorAttachments[0].texture == nil);
    REQUIRE(desc.colorAttachments[1].texture == color_view.RenderTarget());
    REQUIRE(desc.depthAttachment.texture == depth_view.RenderTarget());
    REQUIRE(desc.stencilAttachment.texture == depth_view.RenderTarget());
}

TEST_CASE("Metal component orders compose with guest swizzles", "[video_core][metal]") {
    std::array swizzle{SwizzleSource::R, SwizzleSource::G, SwizzleSource::B, SwizzleSource::A};
    MaxwellToMTL::ApplyComponentOrder(MaxwellToMTL::ComponentOrder::SwapBlueRed, swizzle);
    REQUIRE(swizzle ==
            std::array{SwizzleSource::B, SwizzleSource::G, SwizzleSource::R, SwizzleSource::A});

    swizzle = {SwizzleSource::R, SwizzleSource::G, SwizzleSource::B, SwizzleSource::A};
    MaxwellToMTL::ApplyComponentOrder(MaxwellToMTL::ComponentOrder::SwapSpecial, swizzle);
    REQUIRE(swizzle ==
            std::array{SwizzleSource::A, SwizzleSource::B, SwizzleSource::G, SwizzleSource::R});

    swizzle = {SwizzleSource::R, SwizzleSource::Zero, SwizzleSource::OneFloat, SwizzleSource::A};
    MaxwellToMTL::ApplyComponentOrder(MaxwellToMTL::ComponentOrder::Reverse, swizzle);
    REQUIRE(swizzle == std::array{SwizzleSource::A, SwizzleSource::OneFloat, SwizzleSource::Zero,
                                  SwizzleSource::R});
}
