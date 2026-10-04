// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "maxwell_to_mtl.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <optional>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace Metal {

class Device;

namespace MaxwellToMTL {

using VideoCore::Surface::PixelFormat;

/// How a host format orders the guest format's components. Image views compose this with the
/// guest swizzle so shaders see the guest's component order.
enum class ComponentOrder {
    Identity,
    /// The host format stores red where the guest stores blue, and the other way around.
    SwapBlueRed,
    /// A5B5G5R1 stored as A1BGR5: red and alpha swap, green and blue swap.
    SwapSpecial,
    /// A4B4G4R4 stored as ABGR4: all four components are reversed.
    Reverse,
};

struct FormatInfo {
    MTLPixelFormat format;
    /// Can be a color or depth-stencil render target.
    bool attachable;
    /// Can be written by shaders (storage images).
    bool storage;
    ComponentOrder order;
};

/// Host format for a guest format. Formats with no Metal equivalent return
/// MTLPixelFormatInvalid; formats the device can't sample natively (see IsConverted) return the
/// format the texture cache decodes them to.
[[nodiscard]] FormatInfo SurfaceFormat(const Device& device, PixelFormat format);

/// The guest format is decoded on the CPU before upload (ASTC or BCn without hardware support).
[[nodiscard]] bool IsConverted(const Device& device, PixelFormat format);

/// A guest format with the same memory layout as the data uploaded to the host texture: the
/// format itself, or for converted formats the format they are decoded to. Used for row pitches
/// and block sizes of copies.
[[nodiscard]] PixelFormat HostDataFormat(const Device& device, PixelFormat format);

/// Applies a host component order to a swizzle, like the Vulkan renderer's
/// TryTransformSwizzleIfNeeded.
void ApplyComponentOrder(ComponentOrder order,
                         std::array<Tegra::Texture::SwizzleSource, 4>& swizzle);

[[nodiscard]] MTLTextureSwizzle Swizzle(Tegra::Texture::SwizzleSource source);

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Vertex attribute format, or MTLVertexFormatInvalid when Metal has none. Scaled formats
/// return their integer formats; the shader converts them.
[[nodiscard]] MTLVertexFormat VertexFormat(Maxwell::VertexAttribute::Type type,
                                           Maxwell::VertexAttribute::Size size);

[[nodiscard]] MTLCompareFunction ComparisonOp(Maxwell::ComparisonOp op);

[[nodiscard]] MTLStencilOperation StencilOp(Maxwell::StencilOp::Op op);

[[nodiscard]] MTLBlendOperation BlendEquation(Maxwell::Blend::Equation equation);

[[nodiscard]] MTLBlendFactor BlendFactor(Maxwell::Blend::Factor factor);

[[nodiscard]] MTLWinding FrontFace(Maxwell::FrontFace front_face);

[[nodiscard]] MTLCullMode CullFace(Maxwell::CullFace cull_face);

/// Primitive type a topology draws as, or nullopt when Metal can't draw it. Quads and quad
/// strips draw as triangles after the buffer cache rewrites their indices.
[[nodiscard]] std::optional<MTLPrimitiveType> PrimitiveType(Maxwell::PrimitiveTopology topology);

namespace Sampler {

[[nodiscard]] MTLSamplerMinMagFilter Filter(Tegra::Texture::TextureFilter filter);

[[nodiscard]] MTLSamplerMipFilter MipFilter(Tegra::Texture::TextureMipmapFilter filter);

[[nodiscard]] MTLSamplerAddressMode WrapMode(Tegra::Texture::WrapMode wrap_mode,
                                             Tegra::Texture::TextureFilter filter,
                                             bool is_shadow_map);

[[nodiscard]] MTLCompareFunction DepthCompareFunction(Tegra::Texture::DepthCompareFunc func);

/// Metal only has three border colors; picks the closest.
[[nodiscard]] MTLSamplerBorderColor BorderColor(const std::array<float, 4>& color);

} // namespace Sampler

} // namespace MaxwellToMTL
} // namespace Metal
