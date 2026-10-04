// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <iterator>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_metal/maxwell_to_mtl.h"
#include "video_core/renderer_metal/mtl_device.h"

namespace Metal::MaxwellToMTL {

namespace {

using VideoCore::Surface::IsPixelFormatASTC;
using VideoCore::Surface::IsPixelFormatBCn;
using VideoCore::Surface::IsPixelFormatSRGB;

constexpr u32 Attachable = 1 << 0;
constexpr u32 Storage = 1 << 1;

struct FormatTuple {
    MTLPixelFormat format;
    u32 usage = 0;
    ComponentOrder order = ComponentOrder::Identity;
};

// Metal names packed formats from the least significant bit, Vulkan and Maxwell from the most
// significant, so A2B10G10R10 is RGB10A2 and A1R5G5B5 is BGR5A1.
constexpr FormatTuple FORMAT_TABLE[] = {
    {MTLPixelFormatRGBA8Unorm, Attachable | Storage},                   // A8B8G8R8_UNORM
    {MTLPixelFormatRGBA8Snorm, Attachable | Storage},                   // A8B8G8R8_SNORM
    {MTLPixelFormatRGBA8Sint, Attachable | Storage},                    // A8B8G8R8_SINT
    {MTLPixelFormatRGBA8Uint, Attachable | Storage},                    // A8B8G8R8_UINT
    {MTLPixelFormatB5G6R5Unorm, Attachable},                            // R5G6B5_UNORM
    {MTLPixelFormatB5G6R5Unorm, 0, ComponentOrder::SwapBlueRed},        // B5G6R5_UNORM
    {MTLPixelFormatBGR5A1Unorm, Attachable},                            // A1R5G5B5_UNORM
    {MTLPixelFormatRGB10A2Unorm, Attachable | Storage},                 // A2B10G10R10_UNORM
    {MTLPixelFormatRGB10A2Uint, Attachable | Storage},                  // A2B10G10R10_UINT
    {MTLPixelFormatBGR10A2Unorm, Attachable},                           // A2R10G10B10_UNORM
    {MTLPixelFormatBGR5A1Unorm, Attachable, ComponentOrder::SwapBlueRed}, // A1B5G5R5_UNORM
    {MTLPixelFormatA1BGR5Unorm, 0, ComponentOrder::SwapSpecial},        // A5B5G5R1_UNORM
    {MTLPixelFormatR8Unorm, Attachable | Storage},                      // R8_UNORM
    {MTLPixelFormatR8Snorm, Attachable | Storage},                      // R8_SNORM
    {MTLPixelFormatR8Sint, Attachable | Storage},                       // R8_SINT
    {MTLPixelFormatR8Uint, Attachable | Storage},                       // R8_UINT
    {MTLPixelFormatRGBA16Float, Attachable | Storage},                  // R16G16B16A16_FLOAT
    {MTLPixelFormatRGBA16Unorm, Attachable | Storage},                  // R16G16B16A16_UNORM
    {MTLPixelFormatRGBA16Snorm, Attachable | Storage},                  // R16G16B16A16_SNORM
    {MTLPixelFormatRGBA16Sint, Attachable | Storage},                   // R16G16B16A16_SINT
    {MTLPixelFormatRGBA16Uint, Attachable | Storage},                   // R16G16B16A16_UINT
    {MTLPixelFormatRG11B10Float, Attachable | Storage},                 // B10G11R11_FLOAT
    {MTLPixelFormatRGBA32Uint, Attachable | Storage},                   // R32G32B32A32_UINT
    {MTLPixelFormatBC1_RGBA},                                           // BC1_RGBA_UNORM
    {MTLPixelFormatBC2_RGBA},                                           // BC2_UNORM
    {MTLPixelFormatBC3_RGBA},                                           // BC3_UNORM
    {MTLPixelFormatBC4_RUnorm},                                         // BC4_UNORM
    {MTLPixelFormatBC4_RSnorm},                                         // BC4_SNORM
    {MTLPixelFormatBC5_RGUnorm},                                        // BC5_UNORM
    {MTLPixelFormatBC5_RGSnorm},                                        // BC5_SNORM
    {MTLPixelFormatBC7_RGBAUnorm},                                      // BC7_UNORM
    {MTLPixelFormatBC6H_RGBUfloat},                                     // BC6H_UFLOAT
    {MTLPixelFormatBC6H_RGBFloat},                                      // BC6H_SFLOAT
    {MTLPixelFormatASTC_4x4_LDR},                                       // ASTC_2D_4X4_UNORM
    {MTLPixelFormatBGRA8Unorm, Attachable | Storage},                   // B8G8R8A8_UNORM
    {MTLPixelFormatRGBA32Float, Attachable | Storage},                  // R32G32B32A32_FLOAT
    {MTLPixelFormatRGBA32Sint, Attachable | Storage},                   // R32G32B32A32_SINT
    {MTLPixelFormatRG32Float, Attachable | Storage},                    // R32G32_FLOAT
    {MTLPixelFormatRG32Sint, Attachable | Storage},                     // R32G32_SINT
    {MTLPixelFormatR32Float, Attachable | Storage},                     // R32_FLOAT
    {MTLPixelFormatR16Float, Attachable | Storage},                     // R16_FLOAT
    {MTLPixelFormatR16Unorm, Attachable | Storage},                     // R16_UNORM
    {MTLPixelFormatR16Snorm, Attachable | Storage},                     // R16_SNORM
    {MTLPixelFormatR16Uint, Attachable | Storage},                      // R16_UINT
    {MTLPixelFormatR16Sint, Attachable | Storage},                      // R16_SINT
    {MTLPixelFormatRG16Unorm, Attachable | Storage},                    // R16G16_UNORM
    {MTLPixelFormatRG16Float, Attachable | Storage},                    // R16G16_FLOAT
    {MTLPixelFormatRG16Uint, Attachable | Storage},                     // R16G16_UINT
    {MTLPixelFormatRG16Sint, Attachable | Storage},                     // R16G16_SINT
    {MTLPixelFormatRG16Snorm, Attachable | Storage},                    // R16G16_SNORM
    {MTLPixelFormatInvalid},                                            // R32G32B32_FLOAT
    {MTLPixelFormatRGBA8Unorm_sRGB, Attachable},                        // A8B8G8R8_SRGB
    {MTLPixelFormatRG8Unorm, Attachable | Storage},                     // R8G8_UNORM
    {MTLPixelFormatRG8Snorm, Attachable | Storage},                     // R8G8_SNORM
    {MTLPixelFormatRG8Sint, Attachable | Storage},                      // R8G8_SINT
    {MTLPixelFormatRG8Uint, Attachable | Storage},                      // R8G8_UINT
    {MTLPixelFormatRG32Uint, Attachable | Storage},                     // R32G32_UINT
    {MTLPixelFormatRGBA16Float, Attachable | Storage},                  // R16G16B16X16_FLOAT
    {MTLPixelFormatR32Uint, Attachable | Storage},                      // R32_UINT
    {MTLPixelFormatR32Sint, Attachable | Storage},                      // R32_SINT
    {MTLPixelFormatASTC_8x8_LDR},                                       // ASTC_2D_8X8_UNORM
    {MTLPixelFormatASTC_8x5_LDR},                                       // ASTC_2D_8X5_UNORM
    {MTLPixelFormatASTC_5x4_LDR},                                       // ASTC_2D_5X4_UNORM
    {MTLPixelFormatBGRA8Unorm_sRGB, Attachable},                        // B8G8R8A8_SRGB
    {MTLPixelFormatBC1_RGBA_sRGB},                                      // BC1_RGBA_SRGB
    {MTLPixelFormatBC2_RGBA_sRGB},                                      // BC2_SRGB
    {MTLPixelFormatBC3_RGBA_sRGB},                                      // BC3_SRGB
    {MTLPixelFormatBC7_RGBAUnorm_sRGB},                                 // BC7_SRGB
    {MTLPixelFormatABGR4Unorm, 0, ComponentOrder::Reverse},             // A4B4G4R4_UNORM
    {MTLPixelFormatInvalid},                                            // G4R4_UNORM
    {MTLPixelFormatASTC_4x4_sRGB},                                      // ASTC_2D_4X4_SRGB
    {MTLPixelFormatASTC_8x8_sRGB},                                      // ASTC_2D_8X8_SRGB
    {MTLPixelFormatASTC_8x5_sRGB},                                      // ASTC_2D_8X5_SRGB
    {MTLPixelFormatASTC_5x4_sRGB},                                      // ASTC_2D_5X4_SRGB
    {MTLPixelFormatASTC_5x5_LDR},                                       // ASTC_2D_5X5_UNORM
    {MTLPixelFormatASTC_5x5_sRGB},                                      // ASTC_2D_5X5_SRGB
    {MTLPixelFormatASTC_10x8_LDR},                                      // ASTC_2D_10X8_UNORM
    {MTLPixelFormatASTC_10x8_sRGB},                                     // ASTC_2D_10X8_SRGB
    {MTLPixelFormatASTC_6x6_LDR},                                       // ASTC_2D_6X6_UNORM
    {MTLPixelFormatASTC_6x6_sRGB},                                      // ASTC_2D_6X6_SRGB
    {MTLPixelFormatASTC_10x6_LDR},                                      // ASTC_2D_10X6_UNORM
    {MTLPixelFormatASTC_10x6_sRGB},                                     // ASTC_2D_10X6_SRGB
    {MTLPixelFormatASTC_10x5_LDR},                                      // ASTC_2D_10X5_UNORM
    {MTLPixelFormatASTC_10x5_sRGB},                                     // ASTC_2D_10X5_SRGB
    {MTLPixelFormatASTC_10x10_LDR},                                     // ASTC_2D_10X10_UNORM
    {MTLPixelFormatASTC_10x10_sRGB},                                    // ASTC_2D_10X10_SRGB
    {MTLPixelFormatASTC_12x10_LDR},                                     // ASTC_2D_12X10_UNORM
    {MTLPixelFormatASTC_12x10_sRGB},                                    // ASTC_2D_12X10_SRGB
    {MTLPixelFormatASTC_12x12_LDR},                                     // ASTC_2D_12X12_UNORM
    {MTLPixelFormatASTC_12x12_sRGB},                                    // ASTC_2D_12X12_SRGB
    {MTLPixelFormatASTC_8x6_LDR},                                       // ASTC_2D_8X6_UNORM
    {MTLPixelFormatASTC_8x6_sRGB},                                      // ASTC_2D_8X6_SRGB
    {MTLPixelFormatASTC_6x5_LDR},                                       // ASTC_2D_6X5_UNORM
    {MTLPixelFormatASTC_6x5_sRGB},                                      // ASTC_2D_6X5_SRGB
    {MTLPixelFormatRGB9E5Float},                                        // E5B9G9R9_FLOAT
    {MTLPixelFormatETC2_RGB8},                                          // ETC2_RGB_UNORM
    {MTLPixelFormatEAC_RGBA8},                                          // ETC2_RGBA_UNORM
    {MTLPixelFormatETC2_RGB8A1},                                        // ETC2_RGB_PTA_UNORM
    {MTLPixelFormatETC2_RGB8_sRGB},                                     // ETC2_RGB_SRGB
    {MTLPixelFormatEAC_RGBA8_sRGB},                                     // ETC2_RGBA_SRGB
    {MTLPixelFormatETC2_RGB8A1_sRGB},                                   // ETC2_RGB_PTA_SRGB

    // Depth formats. Apple GPUs have no 24-bit depth, so it is stored as 32-bit float.
    {MTLPixelFormatDepth32Float, Attachable}, // D32_FLOAT
    {MTLPixelFormatDepth16Unorm, Attachable}, // D16_UNORM
    {MTLPixelFormatDepth32Float, Attachable}, // X8_D24_UNORM

    // Stencil formats
    {MTLPixelFormatStencil8, Attachable}, // S8_UINT

    // DepthStencil formats
    {MTLPixelFormatDepth32Float_Stencil8, Attachable}, // D24_UNORM_S8_UINT
    {MTLPixelFormatDepth32Float_Stencil8, Attachable}, // S8_UINT_D24_UNORM
    {MTLPixelFormatDepth32Float_Stencil8, Attachable}, // D32_FLOAT_S8_UINT
};
static_assert(std::size(FORMAT_TABLE) == VideoCore::Surface::MaxPixelFormat);

bool IsAstcConverted(const Device& device, PixelFormat format) {
    return IsPixelFormatASTC(format) && !device.SupportsASTC();
}

bool IsBcnConverted(const Device& device, PixelFormat format) {
    return IsPixelFormatBCn(format) && !device.SupportsBCTextureCompression();
}

} // Anonymous namespace

bool IsConverted(const Device& device, PixelFormat format) {
    return IsAstcConverted(device, format) || IsBcnConverted(device, format);
}

PixelFormat HostDataFormat(const Device& device, PixelFormat format) {
    const bool is_srgb = IsPixelFormatSRGB(format);
    if (IsAstcConverted(device, format)) {
        switch (Settings::values.astc_recompression.GetValue()) {
        case Settings::AstcRecompression::Bc1:
            return is_srgb ? PixelFormat::BC1_RGBA_SRGB : PixelFormat::BC1_RGBA_UNORM;
        case Settings::AstcRecompression::Bc3:
            return is_srgb ? PixelFormat::BC3_SRGB : PixelFormat::BC3_UNORM;
        case Settings::AstcRecompression::Uncompressed:
        default:
            return is_srgb ? PixelFormat::A8B8G8R8_SRGB : PixelFormat::A8B8G8R8_UNORM;
        }
    }
    if (IsBcnConverted(device, format)) {
        // Matches the outputs of VideoCommon::DecompressBCn.
        switch (format) {
        case PixelFormat::BC4_UNORM:
            return PixelFormat::R8_UNORM;
        case PixelFormat::BC4_SNORM:
            return PixelFormat::R8_SNORM;
        case PixelFormat::BC5_UNORM:
            return PixelFormat::R8G8_UNORM;
        case PixelFormat::BC5_SNORM:
            return PixelFormat::R8G8_SNORM;
        case PixelFormat::BC6H_UFLOAT:
        case PixelFormat::BC6H_SFLOAT:
            return PixelFormat::R16G16B16A16_FLOAT;
        default:
            return is_srgb ? PixelFormat::A8B8G8R8_SRGB : PixelFormat::A8B8G8R8_UNORM;
        }
    }
    return format;
}

FormatInfo SurfaceFormat(const Device& device, PixelFormat format) {
    const size_t index = static_cast<size_t>(format);
    ASSERT_MSG(index < std::size(FORMAT_TABLE), "Invalid pixel format={}", index);
    if (index >= std::size(FORMAT_TABLE)) {
        return {MTLPixelFormatInvalid, false, false, ComponentOrder::Identity};
    }
    const PixelFormat data_format = HostDataFormat(device, format);
    // Converted formats take every property from the format they are decoded to.
    const FormatTuple& tuple = FORMAT_TABLE[static_cast<size_t>(data_format)];
    return {
        .format = tuple.format,
        .attachable = (tuple.usage & Attachable) != 0,
        .storage = (tuple.usage & Storage) != 0,
        .order = tuple.order,
    };
}

void ApplyComponentOrder(ComponentOrder order,
                         std::array<Tegra::Texture::SwizzleSource, 4>& swizzle) {
    using Tegra::Texture::SwizzleSource;
    const auto remap = [&swizzle](auto&& func) {
        std::ranges::transform(swizzle, swizzle.begin(), func);
    };
    switch (order) {
    case ComponentOrder::Identity:
        break;
    case ComponentOrder::SwapBlueRed:
        remap([](SwizzleSource value) {
            switch (value) {
            case SwizzleSource::R:
                return SwizzleSource::B;
            case SwizzleSource::B:
                return SwizzleSource::R;
            default:
                return value;
            }
        });
        break;
    case ComponentOrder::SwapSpecial:
        remap([](SwizzleSource value) {
            switch (value) {
            case SwizzleSource::A:
                return SwizzleSource::R;
            case SwizzleSource::R:
                return SwizzleSource::A;
            case SwizzleSource::G:
                return SwizzleSource::B;
            case SwizzleSource::B:
                return SwizzleSource::G;
            default:
                return value;
            }
        });
        break;
    case ComponentOrder::Reverse:
        std::ranges::reverse(swizzle);
        break;
    }
}

MTLTextureSwizzle Swizzle(Tegra::Texture::SwizzleSource source) {
    using Tegra::Texture::SwizzleSource;
    switch (source) {
    case SwizzleSource::Zero:
        return MTLTextureSwizzleZero;
    case SwizzleSource::R:
        return MTLTextureSwizzleRed;
    case SwizzleSource::G:
        return MTLTextureSwizzleGreen;
    case SwizzleSource::B:
        return MTLTextureSwizzleBlue;
    case SwizzleSource::A:
        return MTLTextureSwizzleAlpha;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return MTLTextureSwizzleOne;
    }
    ASSERT_MSG(false, "Invalid swizzle={}", static_cast<u32>(source));
    return MTLTextureSwizzleZero;
}

namespace Sampler {

MTLSamplerMinMagFilter Filter(Tegra::Texture::TextureFilter filter) {
    switch (filter) {
    case Tegra::Texture::TextureFilter::Nearest:
        return MTLSamplerMinMagFilterNearest;
    case Tegra::Texture::TextureFilter::Linear:
        return MTLSamplerMinMagFilterLinear;
    }
    ASSERT_MSG(false, "Invalid sampler filter={}", static_cast<u32>(filter));
    return MTLSamplerMinMagFilterNearest;
}

MTLSamplerMipFilter MipFilter(Tegra::Texture::TextureMipmapFilter filter) {
    switch (filter) {
    case Tegra::Texture::TextureMipmapFilter::None:
        return MTLSamplerMipFilterNotMipmapped;
    case Tegra::Texture::TextureMipmapFilter::Nearest:
        return MTLSamplerMipFilterNearest;
    case Tegra::Texture::TextureMipmapFilter::Linear:
        return MTLSamplerMipFilterLinear;
    }
    ASSERT_MSG(false, "Invalid sampler mipmap filter={}", static_cast<u32>(filter));
    return MTLSamplerMipFilterNotMipmapped;
}

MTLSamplerAddressMode WrapMode(Tegra::Texture::WrapMode wrap_mode,
                               Tegra::Texture::TextureFilter filter, bool is_shadow_map) {
    using Tegra::Texture::WrapMode;
    switch (wrap_mode) {
    case WrapMode::Wrap:
        return MTLSamplerAddressModeRepeat;
    case WrapMode::Mirror:
        return MTLSamplerAddressModeMirrorRepeat;
    case WrapMode::ClampToEdge:
        // Like the Vulkan renderer: edge clamping makes square artifacts in shadow maps.
        return is_shadow_map ? MTLSamplerAddressModeClampToBorderColor
                             : MTLSamplerAddressModeClampToEdge;
    case WrapMode::Border:
        return MTLSamplerAddressModeClampToBorderColor;
    case WrapMode::Clamp:
        // GL_CLAMP: clamps to the edge with nearest filtering and blends with the border with
        // linear filtering.
        if (is_shadow_map || filter == Tegra::Texture::TextureFilter::Linear) {
            return MTLSamplerAddressModeClampToBorderColor;
        }
        return MTLSamplerAddressModeClampToEdge;
    case WrapMode::MirrorOnceClampToEdge:
    case WrapMode::MirrorOnceBorder:
    case WrapMode::MirrorOnceClampOGL:
        return MTLSamplerAddressModeMirrorClampToEdge;
    }
    LOG_WARNING(Render_Metal, "Unimplemented wrap mode={}", static_cast<u32>(wrap_mode));
    return MTLSamplerAddressModeRepeat;
}

MTLCompareFunction DepthCompareFunction(Tegra::Texture::DepthCompareFunc func) {
    using Tegra::Texture::DepthCompareFunc;
    switch (func) {
    case DepthCompareFunc::Never:
        return MTLCompareFunctionNever;
    case DepthCompareFunc::Less:
        return MTLCompareFunctionLess;
    case DepthCompareFunc::Equal:
        return MTLCompareFunctionEqual;
    case DepthCompareFunc::LessEqual:
        return MTLCompareFunctionLessEqual;
    case DepthCompareFunc::Greater:
        return MTLCompareFunctionGreater;
    case DepthCompareFunc::NotEqual:
        return MTLCompareFunctionNotEqual;
    case DepthCompareFunc::GreaterEqual:
        return MTLCompareFunctionGreaterEqual;
    case DepthCompareFunc::Always:
        return MTLCompareFunctionAlways;
    }
    ASSERT_MSG(false, "Invalid depth compare function={}", static_cast<u32>(func));
    return MTLCompareFunctionAlways;
}

MTLSamplerBorderColor BorderColor(const std::array<float, 4>& color) {
    if (color == std::array<float, 4>{0, 0, 0, 0}) {
        return MTLSamplerBorderColorTransparentBlack;
    }
    if (color == std::array<float, 4>{0, 0, 0, 1}) {
        return MTLSamplerBorderColorOpaqueBlack;
    }
    if (color == std::array<float, 4>{1, 1, 1, 1}) {
        return MTLSamplerBorderColorOpaqueWhite;
    }
    // Same approximation as the Vulkan renderer without custom border colors.
    if (color[0] + color[1] + color[2] > 1.35f) {
        return MTLSamplerBorderColorOpaqueWhite;
    }
    return color[3] > 0.5f ? MTLSamplerBorderColorOpaqueBlack
                           : MTLSamplerBorderColorTransparentBlack;
}

} // namespace Sampler

} // namespace Metal::MaxwellToMTL
