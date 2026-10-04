// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/renderer_metal/maxwell_to_mtl.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/util.h"

namespace Metal {

namespace {

using Tegra::Engines::Fermi2D;
using Tegra::Texture::SwizzleSource;
using VideoCommon::BufferImageCopy;
using VideoCommon::ImageCopy;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewType;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::DefaultBlockHeight;
using VideoCore::Surface::DefaultBlockWidth;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::SurfaceType;

// Converts between guest depth/stencil texels and Metal's 32-bit float depth plus separate
// 8-bit stencil. Layouts: 0 X8_D24 (depth in the low 24 bits), 1 D24S8 (depth in the low 24
// bits, stencil in the high 8), 2 S8D24 (stencil in the low 8 bits, depth in the high 24),
// 3 D32F_S8 (a float, then a word with stencil in the low 8 bits).
constexpr const char* DEPTH_STENCIL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct DepthStencilParams {
    uint layout;
    uint row_texels;   // texels per row of the guest buffer
    uint image_texels; // texels per slice of the guest buffer
    uint width;
    uint height;
    uint depth;
};

// precise::divide and rint make 24-bit values round-trip exactly under fast math.
constant float UNORM24_MAX = 16777215.0f;

kernel void unpack_depth_stencil(device const uint* guest [[buffer(0)]],
                                 device float* depth_out [[buffer(1)]],
                                 device uchar* stencil_out [[buffer(2)]],
                                 constant DepthStencilParams& p [[buffer(3)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.depth) {
        return;
    }
    const uint src = gid.z * p.image_texels + gid.y * p.row_texels + gid.x;
    const uint dst = (gid.z * p.height + gid.y) * p.width + gid.x;
    float depth_value;
    uint stencil = 0;
    if (p.layout == 3) {
        depth_value = as_type<float>(guest[src * 2]);
        stencil = guest[src * 2 + 1] & 0xFFu;
    } else {
        const uint word = guest[src];
        if (p.layout == 2) {
            depth_value = precise::divide(float(word >> 8), UNORM24_MAX);
            stencil = word & 0xFFu;
        } else {
            depth_value = precise::divide(float(word & 0xFFFFFFu), UNORM24_MAX);
            stencil = word >> 24;
        }
    }
    depth_out[dst] = depth_value;
    if (p.layout != 0) {
        stencil_out[dst] = uchar(stencil);
    }
}

kernel void pack_depth_stencil(device uint* guest [[buffer(0)]],
                               device const float* depth_in [[buffer(1)]],
                               device const uchar* stencil_in [[buffer(2)]],
                               constant DepthStencilParams& p [[buffer(3)]],
                               uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.depth) {
        return;
    }
    const uint dst = gid.z * p.image_texels + gid.y * p.row_texels + gid.x;
    const uint src = (gid.z * p.height + gid.y) * p.width + gid.x;
    const float depth_value = depth_in[src];
    const uint stencil = p.layout != 0 ? uint(stencil_in[src]) : 0u;
    if (p.layout == 3) {
        guest[dst * 2] = as_type<uint>(depth_value);
        guest[dst * 2 + 1] = stencil;
        return;
    }
    const uint unorm = uint(rint(clamp(depth_value, 0.0f, 1.0f) * UNORM24_MAX));
    if (p.layout == 2) {
        guest[dst] = (unorm << 8) | stencil;
    } else {
        guest[dst] = unorm | (stencil << 24);
    }
}
)";

struct DepthStencilParams {
    u32 layout;
    u32 row_texels;
    u32 image_texels;
    u32 width;
    u32 height;
    u32 depth;
};

u32 DepthStencilLayout(PixelFormat format) {
    switch (format) {
    case PixelFormat::X8_D24_UNORM:
        return 0;
    case PixelFormat::D24_UNORM_S8_UINT:
        return 1;
    case PixelFormat::S8_UINT_D24_UNORM:
        return 2;
    default:
        return 3;
    }
}

constexpr const char* BLIT_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
    float4 src_rect; // u0, v0, u1, v1
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// One triangle covering the viewport; uv spans src_rect across it.
vertex VertexOut blit_vertex(uint vertex_id [[vertex_id]], constant BlitParams& p [[buffer(0)]]) {
    const float2 t = float2((vertex_id << 1) & 2, vertex_id & 2);
    VertexOut out;
    out.position = float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    out.uv = mix(p.src_rect.xy, p.src_rect.zw, t);
    return out;
}

fragment float4 blit_float(VertexOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                           sampler src_sampler [[sampler(0)]]) {
    return src.sample(src_sampler, in.uv, level(0));
}

template <typename T>
static vec<T, 4> read_texel(texture2d<T> src, float2 uv) {
    const uint2 size = uint2(src.get_width(), src.get_height());
    const uint2 coord = min(uint2(max(uv, float2(0.0)) * float2(size)), size - uint2(1));
    return src.read(coord, 0);
}

fragment uint4 blit_uint(VertexOut in [[stage_in]], texture2d<uint> src [[texture(0)]]) {
    return read_texel(src, in.uv);
}

fragment int4 blit_sint(VertexOut in [[stage_in]], texture2d<int> src [[texture(0)]]) {
    return read_texel(src, in.uv);
}
)";

enum BlitKind : u32 {
    BLIT_FLOAT = 0,
    BLIT_UINT = 1,
    BLIT_SINT = 2,
};

struct BlockInfo {
    u32 width;
    u32 height;
    u32 bytes;
};

BlockInfo GetBlockInfo(PixelFormat data_format) {
    return {
        .width = DefaultBlockWidth(data_format),
        .height = DefaultBlockHeight(data_format),
        .bytes = BytesPerBlock(data_format),
    };
}

/// Whether texels can move between the texture and a buffer as they are laid out in guest
/// memory with a plain blit. 24-bit depth is stored as 32-bit float and Metal copies depth and
/// stencil separately, so those formats go through UploadDepthStencil and DownloadDepthStencil.
bool IsBufferCopyable(PixelFormat data_format) {
    switch (data_format) {
    case PixelFormat::Invalid:
    case PixelFormat::R32G32B32_FLOAT:
    case PixelFormat::G4R4_UNORM:
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT:
        return false;
    default:
        return true;
    }
}

void LogUncopyable(PixelFormat format) {
    static std::array<bool, VideoCore::Surface::MaxPixelFormat + 1> logged{};
    const size_t index = std::min(static_cast<size_t>(format), logged.size() - 1);
    if (!logged[index]) {
        logged[index] = true;
        LOG_WARNING(Render_Metal, "Copies of {} images to and from memory are not implemented",
                    static_cast<u32>(format));
    }
}

MTLTextureType TextureType(const ImageInfo& info) {
    switch (info.type) {
    case ImageType::e3D:
        return MTLTextureType3D;
    case ImageType::e1D:
    case ImageType::e2D:
    case ImageType::Linear:
    case ImageType::Buffer:
        break;
    }
    // 1D images are 2D textures one texel high: Metal 1D textures can't have mipmaps or be
    // rendered to.
    if (info.num_samples > 1) {
        return info.resources.layers > 1 ? MTLTextureType2DMultisampleArray
                                         : MTLTextureType2DMultisample;
    }
    return info.resources.layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
}

/// Metal wants a zero image stride for copies to and from textures without slices.
NSUInteger BytesPerImageArgument(id<MTLTexture> texture, size_t image_bytes) {
    switch (texture.textureType) {
    case MTLTextureType3D:
    case MTLTextureType2DArray:
    case MTLTextureTypeCube:
    case MTLTextureTypeCubeArray:
        return image_bytes;
    default:
        return 0;
    }
}

NSUInteger MipSize(NSUInteger size, s32 level) {
    return std::max<NSUInteger>(size >> level, 1);
}

/// Calls func(src_slice, src_z, dst_slice, dst_z, depth) for each slice-aligned piece of a copy,
/// so copies between 3D textures and array layers line up.
template <typename Func>
void ForEachCopySlice(const ImageCopy& copy, bool src_3d, bool dst_3d, Func&& func) {
    if (src_3d && dst_3d) {
        func(0U, static_cast<u32>(copy.src_offset.z), 0U, static_cast<u32>(copy.dst_offset.z),
             copy.extent.depth);
        return;
    }
    const u32 count = std::max<u32>(src_3d ? copy.extent.depth : copy.src_subresource.num_layers,
                                    dst_3d ? copy.extent.depth : copy.dst_subresource.num_layers);
    for (u32 i = 0; i < count; ++i) {
        const u32 src_slice = src_3d ? 0 : static_cast<u32>(copy.src_subresource.base_layer) + i;
        const u32 src_z = src_3d ? static_cast<u32>(copy.src_offset.z) + i : 0;
        const u32 dst_slice = dst_3d ? 0 : static_cast<u32>(copy.dst_subresource.base_layer) + i;
        const u32 dst_z = dst_3d ? static_cast<u32>(copy.dst_offset.z) + i : 0;
        func(src_slice, src_z, dst_slice, dst_z, 1U);
    }
}

id<MTLTexture> NewNullTexture(id<MTLDevice> device, MTLTextureType type, NSUInteger slices) {
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = type;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = slices;
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    const u32 zero = 0;
    for (NSUInteger slice = 0; slice < slices; ++slice) {
        [texture replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
                   mipmapLevel:0
                         slice:slice
                     withBytes:&zero
                   bytesPerRow:sizeof(zero)
                 bytesPerImage:sizeof(zero)];
    }
    texture.label = @"null image";
    return texture;
}

/// For depth-stencil formats the swizzle picks the aspect a texture view samples: red selects
/// depth (stencil for S8_UINT_D24_UNORM).
bool SamplesStencil(const VideoCommon::ImageViewInfo& info) {
    const auto swizzle = info.Swizzle();
    const bool any_r =
        std::ranges::any_of(swizzle, [](SwizzleSource s) { return s == SwizzleSource::R; });
    switch (info.format) {
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::D32_FLOAT_S8_UINT:
        return !any_r;
    case PixelFormat::S8_UINT_D24_UNORM:
        return any_r;
    case PixelFormat::S8_UINT:
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                         StagingBufferPool& staging_buffer_pool_)
    : device{device_}, scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_} {
    MTLSamplerDescriptor* desc = [[MTLSamplerDescriptor alloc] init];
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.magFilter = MTLSamplerMinMagFilterNearest;
    nearest_sampler = [device.GetDevice() newSamplerStateWithDescriptor:desc];
    desc.minFilter = MTLSamplerMinMagFilterLinear;
    desc.magFilter = MTLSamplerMinMagFilterLinear;
    linear_sampler = [device.GetDevice() newSamplerStateWithDescriptor:desc];
}

TextureCacheRuntime::~TextureCacheRuntime() = default;

void TextureCacheRuntime::Finish() {
    scheduler.Finish();
}

StagingBufferRef TextureCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_buffer_pool.Request(size, MemoryUsage::Upload);
}

StagingBufferRef TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_buffer_pool.Request(size, MemoryUsage::Download, deferred);
}

void TextureCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_buffer_pool.FreeDeferred(ref);
}

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    return device.GetRecommendedWorkingSetSize();
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetCurrentAllocatedSize();
}

void TextureCacheRuntime::BlitImage(Framebuffer*, ImageView& dst, ImageView& src,
                                    const Region2D& dst_region, const Region2D& src_region,
                                    Fermi2D::Filter filter, Fermi2D::Operation operation) {
    if (dst.ImageHandle() == nil || src.ImageHandle() == nil) {
        return;
    }
    if (operation != Fermi2D::Operation::SrcCopy) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "Blit operation {} is drawn as a copy",
                        static_cast<u32>(operation));
        }
    }
    const s32 src_width = src_region.end.x - src_region.start.x;
    const s32 src_height = src_region.end.y - src_region.start.y;
    const s32 dst_width = dst_region.end.x - dst_region.start.x;
    const s32 dst_height = dst_region.end.y - dst_region.start.y;
    if (src_width == dst_width && src_height == dst_height && src_width > 0 && src_height > 0 &&
        src.HostFormat() == dst.HostFormat() && src.Samples() == dst.Samples()) {
        // Unscaled, unflipped and the same format: a plain copy.
        const ImageCopy copy{
            .src_subresource{
                .base_level = src.range.base.level,
                .base_layer = src.range.base.layer,
                .num_layers = 1,
            },
            .dst_subresource{
                .base_level = dst.range.base.level,
                .base_layer = dst.range.base.layer,
                .num_layers = 1,
            },
            .src_offset{src_region.start.x, src_region.start.y, 0},
            .dst_offset{dst_region.start.x, dst_region.start.y, 0},
            .extent{static_cast<u32>(src_width), static_cast<u32>(src_height), 1},
        };
        BlitCopy(dst.CopyTarget(), src.CopyTarget(), std::span{&copy, 1});
        return;
    }
    if (GetFormatType(dst.format) != SurfaceType::ColorTexture ||
        GetFormatType(src.format) != SurfaceType::ColorTexture || src.Samples() > 1 ||
        dst.Samples() > 1) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal,
                        "Scaled or converting blits of depth or multisampled images are not "
                        "implemented");
        }
        return;
    }
    DrawBlit(dst, src, dst_region, src_region, filter == Fermi2D::Filter::Bilinear);
}

void TextureCacheRuntime::DrawBlit(ImageView& dst, ImageView& src, const Region2D& dst_region,
                                   const Region2D& src_region, bool linear) {
    id<MTLTexture> src_texture = src.Handle(Shader::TextureType::Color2D);
    id<MTLTexture> dst_texture = dst.RenderTarget();
    if (src_texture == nil || dst_texture == nil) {
        return;
    }
    if ((dst.ImageHandle().usage & MTLTextureUsageRenderTarget) == 0) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "Scaled blits to format {} are not implemented",
                        static_cast<u32>(dst.format));
        }
        return;
    }
    u32 kind = BLIT_FLOAT;
    if (VideoCore::Surface::IsPixelFormatInteger(dst.format)) {
        kind = VideoCore::Surface::IsPixelFormatSignedInteger(dst.format) ? BLIT_SINT : BLIT_UINT;
    }

    // Draw with the destination region's corners in order and flip the source instead.
    float u0 = static_cast<float>(src_region.start.x) / static_cast<float>(src_texture.width);
    float u1 = static_cast<float>(src_region.end.x) / static_cast<float>(src_texture.width);
    float v0 = static_cast<float>(src_region.start.y) / static_cast<float>(src_texture.height);
    float v1 = static_cast<float>(src_region.end.y) / static_cast<float>(src_texture.height);
    if (dst_region.end.x < dst_region.start.x) {
        std::swap(u0, u1);
    }
    if (dst_region.end.y < dst_region.start.y) {
        std::swap(v0, v1);
    }
    const s32 x0 = std::min(dst_region.start.x, dst_region.end.x);
    const s32 x1 = std::max(dst_region.start.x, dst_region.end.x);
    const s32 y0 = std::min(dst_region.start.y, dst_region.end.y);
    const s32 y1 = std::max(dst_region.start.y, dst_region.end.y);
    if (x0 == x1 || y0 == y1) {
        return;
    }
    const std::array<float, 4> src_rect{u0, v0, u1, v1};

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = dst_texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderPipelineState> pipeline = BlitPipeline(dst.HostFormat(), kind);
    scheduler.EndEncoding();
    id<MTLRenderCommandEncoder> encoder =
        [scheduler.CommandBuffer() renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:(MTLViewport){
                             .originX = static_cast<double>(x0),
                             .originY = static_cast<double>(y0),
                             .width = static_cast<double>(x1 - x0),
                             .height = static_cast<double>(y1 - y0),
                             .znear = 0.0,
                             .zfar = 1.0,
                         }];
    [encoder setVertexBytes:src_rect.data() length:sizeof(src_rect) atIndex:0];
    [encoder setFragmentTexture:src_texture atIndex:0];
    [encoder setFragmentSamplerState:linear ? linear_sampler : nearest_sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
}

bool TextureCacheRuntime::IsConvertedDepthStencil(PixelFormat data_format) noexcept {
    switch (data_format) {
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

namespace {

/// One copy's region in the tight depth and stencil buffers the conversion goes through.
struct DepthStencilRegion {
    DepthStencilParams params;
    size_t texels;
    /// Slices: layers of an array texture, or depth of a 3D texture.
    u32 slices;
};

DepthStencilRegion MakeDepthStencilRegion(PixelFormat format, bool is_3d,
                                          const BufferImageCopy& copy) {
    const u32 slices = is_3d ? copy.image_extent.depth
                             : static_cast<u32>(copy.image_subresource.num_layers);
    const DepthStencilParams params{
        .layout = DepthStencilLayout(format),
        .row_texels = copy.buffer_row_length,
        .image_texels = copy.buffer_row_length * copy.buffer_image_height,
        .width = copy.image_extent.width,
        .height = copy.image_extent.height,
        .depth = slices,
    };
    return {
        .params = params,
        .texels = size_t{params.width} * params.height * slices,
        .slices = slices,
    };
}

void DispatchDepthStencil(Scheduler& scheduler, id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> packed, size_t packed_offset, id<MTLBuffer> temp,
                          size_t depth_offset, size_t stencil_offset,
                          const DepthStencilParams& params) {
    id<MTLComputeCommandEncoder> encoder = scheduler.ComputeEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:packed offset:packed_offset atIndex:0];
    [encoder setBuffer:temp offset:depth_offset atIndex:1];
    [encoder setBuffer:temp offset:stencil_offset atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger group_width =
        std::clamp<NSUInteger>(params.width, 1, pipeline.threadExecutionWidth);
    const NSUInteger max_rows =
        std::max<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup / group_width, 1);
    const NSUInteger group_height = std::clamp<NSUInteger>(params.height, 1, max_rows);
    [encoder dispatchThreads:MTLSizeMake(params.width, params.height, params.depth)
        threadsPerThreadgroup:MTLSizeMake(group_width, group_height, 1)];
}

} // Anonymous namespace

void TextureCacheRuntime::UploadDepthStencil(const TextureCopyTarget& dst, id<MTLBuffer> buffer,
                                             size_t offset,
                                             std::span<const BufferImageCopy> copies) {
    const bool has_stencil = dst.data_format != PixelFormat::X8_D24_UNORM;
    id<MTLComputePipelineState> pipeline = DepthStencilPipeline(false);
    for (const BufferImageCopy& copy : copies) {
        const DepthStencilRegion region = MakeDepthStencilRegion(dst.data_format, dst.is_3d, copy);
        if (region.texels == 0) {
            continue;
        }
        // Depth floats, then stencil bytes.
        const size_t depth_bytes = region.texels * sizeof(float);
        const StagingBufferRef temp =
            staging_buffer_pool.Request(depth_bytes + region.texels, MemoryUsage::Download);
        const size_t stencil_offset = temp.offset + depth_bytes;
        DispatchDepthStencil(scheduler, pipeline, buffer, offset + copy.buffer_offset,
                             temp.buffer, temp.offset, stencil_offset, region.params);

        const u32 width = region.params.width;
        const u32 height = region.params.height;
        const size_t slice_texels = size_t{width} * height;
        id<MTLBlitCommandEncoder> encoder = scheduler.BlitEncoder();
        const auto copy_aspect = [&](size_t base, size_t texel_bytes, MTLBlitOption option) {
            for (u32 slice = 0; slice < (dst.is_3d ? 1 : region.slices); ++slice) {
                const size_t row_bytes = width * texel_bytes;
                [encoder copyFromBuffer:temp.buffer
                           sourceOffset:base + slice * slice_texels * texel_bytes
                      sourceBytesPerRow:row_bytes
                    sourceBytesPerImage:dst.is_3d ? row_bytes * height : 0
                             sourceSize:MTLSizeMake(width, height,
                                                    dst.is_3d ? region.slices : 1)
                              toTexture:dst.texture
                       destinationSlice:static_cast<NSUInteger>(
                                            dst.is_3d ? 0
                                                      : copy.image_subresource.base_layer + slice)
                       destinationLevel:static_cast<NSUInteger>(copy.image_subresource.base_level)
                      destinationOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                                      dst.is_3d ? copy.image_offset.z : 0)
                                options:option];
            }
        };
        copy_aspect(temp.offset, sizeof(float),
                    has_stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone);
        if (has_stencil) {
            copy_aspect(stencil_offset, 1, MTLBlitOptionStencilFromDepthStencil);
        }
    }
}

void TextureCacheRuntime::DownloadDepthStencil(const TextureCopyTarget& src, id<MTLBuffer> buffer,
                                               size_t offset,
                                               std::span<const BufferImageCopy> copies) {
    const bool has_stencil = src.data_format != PixelFormat::X8_D24_UNORM;
    id<MTLComputePipelineState> pipeline = DepthStencilPipeline(true);
    for (const BufferImageCopy& copy : copies) {
        const DepthStencilRegion region = MakeDepthStencilRegion(src.data_format, src.is_3d, copy);
        if (region.texels == 0) {
            continue;
        }
        const size_t depth_bytes = region.texels * sizeof(float);
        const StagingBufferRef temp =
            staging_buffer_pool.Request(depth_bytes + region.texels, MemoryUsage::Download);
        const size_t stencil_offset = temp.offset + depth_bytes;

        const u32 width = region.params.width;
        const u32 height = region.params.height;
        const size_t slice_texels = size_t{width} * height;
        id<MTLBlitCommandEncoder> encoder = scheduler.BlitEncoder();
        const auto copy_aspect = [&](size_t base, size_t texel_bytes, MTLBlitOption option) {
            for (u32 slice = 0; slice < (src.is_3d ? 1 : region.slices); ++slice) {
                const size_t row_bytes = width * texel_bytes;
                [encoder copyFromTexture:src.texture
                                 sourceSlice:static_cast<NSUInteger>(
                                                 src.is_3d
                                                     ? 0
                                                     : copy.image_subresource.base_layer + slice)
                                 sourceLevel:static_cast<NSUInteger>(
                                                 copy.image_subresource.base_level)
                                sourceOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                                           src.is_3d ? copy.image_offset.z : 0)
                                  sourceSize:MTLSizeMake(width, height,
                                                         src.is_3d ? region.slices : 1)
                                    toBuffer:temp.buffer
                           destinationOffset:base + slice * slice_texels * texel_bytes
                      destinationBytesPerRow:row_bytes
                    destinationBytesPerImage:src.is_3d ? row_bytes * height : 0
                                     options:option];
            }
        };
        copy_aspect(temp.offset, sizeof(float),
                    has_stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone);
        if (has_stencil) {
            copy_aspect(stencil_offset, 1, MTLBlitOptionStencilFromDepthStencil);
        }
        DispatchDepthStencil(scheduler, pipeline, buffer, offset + copy.buffer_offset,
                             temp.buffer, temp.offset, stencil_offset, region.params);
    }
}

id<MTLComputePipelineState> TextureCacheRuntime::DepthStencilPipeline(bool pack) {
    __strong id<MTLComputePipelineState>& pipeline =
        pack ? pack_depth_stencil_pipeline : unpack_depth_stencil_pipeline;
    if (pipeline != nil) {
        return pipeline;
    }
    NSError* error = nil;
    if (depth_stencil_library == nil) {
        depth_stencil_library =
            [device.GetDevice() newLibraryWithSource:@(DEPTH_STENCIL_SHADER_SOURCE)
                                             options:nil
                                               error:&error];
        if (depth_stencil_library == nil) {
            throw std::runtime_error(
                std::string{"Failed to compile Metal depth/stencil shaders: "} +
                error.localizedDescription.UTF8String);
        }
    }
    id<MTLFunction> function = [depth_stencil_library
        newFunctionWithName:pack ? @"pack_depth_stencil" : @"unpack_depth_stencil"];
    pipeline = [device.GetDevice() newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string{"Failed to create Metal depth/stencil pipeline: "} +
                                 error.localizedDescription.UTF8String);
    }
    return pipeline;
}

id<MTLRenderPipelineState> TextureCacheRuntime::BlitPipeline(MTLPixelFormat format, u32 kind) {
    const u64 key = (static_cast<u64>(format) << 2) | kind;
    if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
        return it->second;
    }
    NSError* error = nil;
    if (blit_library == nil) {
        blit_library = [device.GetDevice() newLibraryWithSource:@(BLIT_SHADER_SOURCE)
                                                        options:nil
                                                          error:&error];
        if (blit_library == nil) {
            throw std::runtime_error(std::string{"Failed to compile Metal blit shaders: "} +
                                     error.localizedDescription.UTF8String);
        }
    }
    NSString* fragment = @"blit_float";
    if (kind == BLIT_UINT) {
        fragment = @"blit_uint";
    } else if (kind == BLIT_SINT) {
        fragment = @"blit_sint";
    }
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = [blit_library newFunctionWithName:@"blit_vertex"];
    desc.fragmentFunction = [blit_library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat = format;
    id<MTLRenderPipelineState> pipeline =
        [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string{"Failed to create Metal blit pipeline: "} +
                                 error.localizedDescription.UTF8String);
    }
    blit_pipelines.emplace(key, pipeline);
    return pipeline;
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src, std::span<const ImageCopy> copies) {
    if (dst.Handle() == nil || src.Handle() == nil) {
        return;
    }
    if (dst.info.num_samples != src.info.num_samples) {
        return CopyImageMSAA(dst, src, copies);
    }
    if (dst.HostFormat() == src.HostFormat()) {
        BlitCopy(dst.CopyTarget(), src.CopyTarget(), copies);
    } else {
        CopyThroughBuffer(dst.CopyTarget(), src.CopyTarget(), copies);
    }
}

void TextureCacheRuntime::CopyImageMSAA(Image&, Image&, std::span<const ImageCopy>) {
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_WARNING(Render_Metal, "Copies between images with different sample counts are not "
                                  "implemented");
    }
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) {
    // Only called when one image is color and the other depth or stencil; Metal blits can't
    // convert between them, so copy the bits through a buffer.
    return dst.info.format != src.info.format;
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const ImageCopy> copies) {
    if (dst.Handle() == nil || src.Handle() == nil) {
        return;
    }
    CopyThroughBuffer(dst.CopyTarget(), src.CopyTarget(), copies);
}

void TextureCacheRuntime::ConvertImage(Framebuffer*, ImageView& dst_view, ImageView& src_view) {
    if (dst_view.ImageHandle() == nil || src_view.ImageHandle() == nil) {
        return;
    }
    const ImageCopy copy{
        .src_subresource{
            .base_level = src_view.range.base.level,
            .base_layer = src_view.range.base.layer,
            .num_layers = 1,
        },
        .dst_subresource{
            .base_level = dst_view.range.base.level,
            .base_layer = dst_view.range.base.layer,
            .num_layers = 1,
        },
        .src_offset{0, 0, 0},
        .dst_offset{0, 0, 0},
        .extent{
            std::min(dst_view.size.width, src_view.size.width),
            std::min(dst_view.size.height, src_view.size.height),
            1,
        },
    };
    CopyThroughBuffer(dst_view.CopyTarget(), src_view.CopyTarget(), std::span{&copy, 1});
}

void TextureCacheRuntime::BlitCopy(const TextureCopyTarget& dst, const TextureCopyTarget& src,
                                   std::span<const ImageCopy> copies) {
    id<MTLBlitCommandEncoder> encoder = scheduler.BlitEncoder();
    for (const ImageCopy& copy : copies) {
        ForEachCopySlice(copy, src.is_3d, dst.is_3d,
                         [&](u32 src_slice, u32 src_z, u32 dst_slice, u32 dst_z, u32 depth) {
                             [encoder copyFromTexture:src.texture
                                          sourceSlice:src_slice
                                          sourceLevel:copy.src_subresource.base_level
                                         sourceOrigin:MTLOriginMake(copy.src_offset.x,
                                                                    copy.src_offset.y, src_z)
                                           sourceSize:MTLSizeMake(copy.extent.width,
                                                                  copy.extent.height, depth)
                                            toTexture:dst.texture
                                     destinationSlice:dst_slice
                                     destinationLevel:copy.dst_subresource.base_level
                                    destinationOrigin:MTLOriginMake(copy.dst_offset.x,
                                                                    copy.dst_offset.y, dst_z)];
                         });
    }
}

void TextureCacheRuntime::CopyThroughBuffer(const TextureCopyTarget& dst,
                                            const TextureCopyTarget& src,
                                            std::span<const ImageCopy> copies) {
    const BlockInfo src_block = GetBlockInfo(src.data_format);
    const BlockInfo dst_block = GetBlockInfo(dst.data_format);
    if (!IsBufferCopyable(src.data_format) || !IsBufferCopyable(dst.data_format) ||
        src_block.bytes != dst_block.bytes || src.texture.sampleCount > 1 ||
        dst.texture.sampleCount > 1) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "Copies from format {} to format {} are not implemented",
                        static_cast<u32>(src.data_format), static_cast<u32>(dst.data_format));
        }
        return;
    }
    for (const ImageCopy& copy : copies) {
        const u32 blocks_x = Common::DivCeil(copy.extent.width, src_block.width);
        const u32 blocks_y = Common::DivCeil(copy.extent.height, src_block.height);
        const size_t row_bytes = size_t{blocks_x} * src_block.bytes;
        const size_t slice_bytes = row_bytes * blocks_y;

        const s32 dst_level = copy.dst_subresource.base_level;
        const NSUInteger dst_width = std::min<NSUInteger>(
            NSUInteger{blocks_x} * dst_block.width,
            MipSize(dst.texture.width, dst_level) - static_cast<NSUInteger>(copy.dst_offset.x));
        const NSUInteger dst_height = std::min<NSUInteger>(
            NSUInteger{blocks_y} * dst_block.height,
            MipSize(dst.texture.height, dst_level) - static_cast<NSUInteger>(copy.dst_offset.y));

        std::vector<std::array<u32, 4>> slices;
        ForEachCopySlice(copy, src.is_3d, dst.is_3d,
                         [&](u32 src_slice, u32 src_z, u32 dst_slice, u32 dst_z, u32 depth) {
                             for (u32 z = 0; z < depth; ++z) {
                                 slices.push_back({src_slice, src_z + z, dst_slice, dst_z + z});
                             }
                         });
        const StagingBufferRef staging =
            staging_buffer_pool.Request(slice_bytes * slices.size(), MemoryUsage::Download);
        id<MTLBlitCommandEncoder> encoder = scheduler.BlitEncoder();
        for (size_t i = 0; i < slices.size(); ++i) {
            const auto [src_slice, src_z, dst_slice, dst_z] = slices[i];
            const size_t offset = staging.offset + i * slice_bytes;
            [encoder copyFromTexture:src.texture
                             sourceSlice:src_slice
                             sourceLevel:copy.src_subresource.base_level
                            sourceOrigin:MTLOriginMake(copy.src_offset.x, copy.src_offset.y, src_z)
                              sourceSize:MTLSizeMake(copy.extent.width, copy.extent.height, 1)
                                toBuffer:staging.buffer
                       destinationOffset:offset
                  destinationBytesPerRow:row_bytes
                destinationBytesPerImage:BytesPerImageArgument(src.texture, slice_bytes)];
            [encoder copyFromBuffer:staging.buffer
                       sourceOffset:offset
                  sourceBytesPerRow:row_bytes
                sourceBytesPerImage:BytesPerImageArgument(dst.texture, slice_bytes)
                         sourceSize:MTLSizeMake(dst_width, dst_height, 1)
                          toTexture:dst.texture
                   destinationSlice:dst_slice
                   destinationLevel:dst_level
                  destinationOrigin:MTLOriginMake(copy.dst_offset.x, copy.dst_offset.y, dst_z)];
        }
    }
}

Image::Image(TextureCacheRuntime& runtime_, const ImageInfo& info_, GPUVAddr gpu_addr_,
             VAddr cpu_addr_)
    : VideoCommon::ImageBase(info_, gpu_addr_, cpu_addr_), runtime{&runtime_} {
    const Device& device = runtime->device;
    data_format = MaxwellToMTL::HostDataFormat(device, info.format);
    if (MaxwellToMTL::IsConverted(device, info.format)) {
        flags |= ImageFlagBits::Converted;
        flags |= ImageFlagBits::CostlyLoad;
    }
    if (info.type == ImageType::Buffer) {
        return;
    }
    const MaxwellToMTL::FormatInfo format_info = MaxwellToMTL::SurfaceFormat(device, info.format);
    host_format = format_info.format;
    if (host_format == MTLPixelFormatInvalid) {
        LOG_ERROR(Render_Metal, "Format {} has no Metal equivalent; using RGBA8",
                  static_cast<u32>(info.format));
        host_format = MTLPixelFormatRGBA8Unorm;
    }
    const auto [samples_x, samples_y] = VideoCommon::SamplesLog2(info.num_samples);
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = TextureType(info);
    desc.pixelFormat = host_format;
    desc.width = info.size.width >> samples_x;
    desc.height = info.type == ImageType::e1D ? 1 : info.size.height >> samples_y;
    desc.depth = info.type == ImageType::e3D ? info.size.depth : 1;
    desc.mipmapLevelCount = static_cast<NSUInteger>(info.resources.levels);
    desc.arrayLength =
        info.type == ImageType::e3D ? 1 : static_cast<NSUInteger>(info.resources.layers);
    desc.sampleCount = info.num_samples;
    desc.storageMode = MTLStorageModePrivate;
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (format_info.attachable) {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (format_info.storage && info.num_samples == 1) {
        usage |= MTLTextureUsageShaderWrite;
    }
    desc.usage = usage;
    texture = [device.GetDevice() newTextureWithDescriptor:desc];
    if (texture == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to create a {}x{}x{} texture of format {}",
                     desc.width, desc.height, desc.depth, static_cast<u32>(info.format));
        throw std::runtime_error("Failed to create a Metal texture");
    }
    texture.label = [NSString stringWithFormat:@"Image 0x%llx",
                                               static_cast<unsigned long long>(gpu_addr)];
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

Image::~Image() = default;

void Image::UploadMemory(id<MTLBuffer> buffer, size_t offset,
                         std::span<const BufferImageCopy> copies) {
    if (texture == nil) {
        return;
    }
    if (TextureCacheRuntime::IsConvertedDepthStencil(data_format)) {
        initialized = true;
        runtime->UploadDepthStencil(CopyTarget(), buffer, offset, copies);
        return;
    }
    if (!IsBufferCopyable(data_format)) {
        LogUncopyable(info.format);
        return;
    }
    initialized = true;
    const BlockInfo block = GetBlockInfo(data_format);
    const bool is_3d = info.type == ImageType::e3D;
    id<MTLBlitCommandEncoder> encoder = runtime->scheduler.BlitEncoder();
    for (const BufferImageCopy& copy : copies) {
        const size_t row_bytes =
            size_t{Common::DivCeil(copy.buffer_row_length, block.width)} * block.bytes;
        const size_t image_bytes =
            row_bytes * Common::DivCeil(copy.buffer_image_height, block.height);
        const u32 depth = is_3d ? copy.image_extent.depth : 1;
        for (s32 layer = 0; layer < (is_3d ? 1 : copy.image_subresource.num_layers); ++layer) {
            [encoder copyFromBuffer:buffer
                       sourceOffset:offset + copy.buffer_offset +
                                    static_cast<size_t>(layer) * image_bytes
                  sourceBytesPerRow:row_bytes
                sourceBytesPerImage:BytesPerImageArgument(texture, image_bytes)
                         sourceSize:MTLSizeMake(copy.image_extent.width,
                                                copy.image_extent.height, depth)
                          toTexture:texture
                   destinationSlice:static_cast<NSUInteger>(copy.image_subresource.base_layer +
                                                            layer)
                   destinationLevel:static_cast<NSUInteger>(copy.image_subresource.base_level)
                  destinationOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                                  is_3d ? copy.image_offset.z : 0)];
        }
    }
}

void Image::UploadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    UploadMemory(map.buffer, map.offset, copies);
}

void Image::DownloadMemory(id<MTLBuffer> buffer, size_t offset,
                           std::span<const BufferImageCopy> copies) {
    std::array buffers{buffer};
    std::array offsets{offset};
    DownloadMemory(buffers, offsets, copies);
}

void Image::DownloadMemory(std::span<id<MTLBuffer>> buffers, std::span<size_t> offsets,
                           std::span<const BufferImageCopy> copies) {
    if (texture == nil) {
        return;
    }
    if (TextureCacheRuntime::IsConvertedDepthStencil(data_format)) {
        for (size_t index = 0; index < buffers.size(); ++index) {
            runtime->DownloadDepthStencil(CopyTarget(), buffers[index], offsets[index], copies);
        }
        return;
    }
    if (!IsBufferCopyable(data_format)) {
        LogUncopyable(info.format);
        return;
    }
    const BlockInfo block = GetBlockInfo(data_format);
    const bool is_3d = info.type == ImageType::e3D;
    id<MTLBlitCommandEncoder> encoder = runtime->scheduler.BlitEncoder();
    for (size_t index = 0; index < buffers.size(); ++index) {
        for (const BufferImageCopy& copy : copies) {
            const size_t row_bytes =
                size_t{Common::DivCeil(copy.buffer_row_length, block.width)} * block.bytes;
            const size_t image_bytes =
                row_bytes * Common::DivCeil(copy.buffer_image_height, block.height);
            const u32 depth = is_3d ? copy.image_extent.depth : 1;
            for (s32 layer = 0; layer < (is_3d ? 1 : copy.image_subresource.num_layers);
                 ++layer) {
                [encoder copyFromTexture:texture
                                 sourceSlice:static_cast<NSUInteger>(
                                                 copy.image_subresource.base_layer + layer)
                                 sourceLevel:static_cast<NSUInteger>(
                                                 copy.image_subresource.base_level)
                                sourceOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                                           is_3d ? copy.image_offset.z : 0)
                                  sourceSize:MTLSizeMake(copy.image_extent.width,
                                                         copy.image_extent.height, depth)
                                    toBuffer:buffers[index]
                           destinationOffset:offsets[index] + copy.buffer_offset +
                                             static_cast<size_t>(layer) * image_bytes
                      destinationBytesPerRow:row_bytes
                    destinationBytesPerImage:BytesPerImageArgument(texture, image_bytes)];
            }
        }
    }
}

void Image::DownloadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    DownloadMemory(map.buffer, map.offset, copies);
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id_, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr},
      copy_target{image.CopyTarget()}, samples{image.info.num_samples} {
    using Shader::TextureType;
    id<MTLTexture> base = image.Handle();
    if (base == nil) {
        return;
    }
    const MaxwellToMTL::FormatInfo format_info =
        MaxwellToMTL::SurfaceFormat(runtime.device, format);
    const bool is_color = GetFormatType(format) == SurfaceType::ColorTexture;
    // Metal can't reinterpret depth formats in views (only as X32_Stencil8, below).
    host_format = format_info.format != MTLPixelFormatInvalid && is_color
                      ? format_info.format
                      : image.HostFormat();

    std::array<SwizzleSource, 4> swizzle{SwizzleSource::R, SwizzleSource::G, SwizzleSource::B,
                                         SwizzleSource::A};
    MTLPixelFormat sample_format = host_format;
    if (!info.IsRenderTarget()) {
        if (is_color) {
            swizzle = info.Swizzle();
            MaxwellToMTL::ApplyComponentOrder(format_info.order, swizzle);
        } else if (SamplesStencil(info) && host_format == MTLPixelFormatDepth32Float_Stencil8) {
            sample_format = MTLPixelFormatX32_Stencil8;
        }
    }
    const bool identity_swizzle =
        swizzle == std::array{SwizzleSource::R, SwizzleSource::G, SwizzleSource::B,
                              SwizzleSource::A};
    const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        MaxwellToMTL::Swizzle(swizzle[0]), MaxwellToMTL::Swizzle(swizzle[1]),
        MaxwellToMTL::Swizzle(swizzle[2]), MaxwellToMTL::Swizzle(swizzle[3]));

    const NSRange levels = NSMakeRange(static_cast<NSUInteger>(range.base.level),
                                       static_cast<NSUInteger>(range.extent.levels));
    const bool is_3d = image.info.type == ImageType::e3D;
    const NSRange all_slices =
        is_3d ? NSMakeRange(0, 1)
              : NSMakeRange(static_cast<NSUInteger>(range.base.layer),
                            static_cast<NSUInteger>(range.extent.layers));
    const auto make = [&](MTLTextureType type, NSUInteger num_slices, MTLPixelFormat view_format,
                          bool swizzled) -> id<MTLTexture> {
        const NSRange slices = NSMakeRange(all_slices.location, num_slices);
        if (swizzled && !identity_swizzle) {
            return [base newTextureViewWithPixelFormat:view_format
                                           textureType:type
                                                levels:levels
                                                slices:slices
                                               swizzle:channels];
        }
        return [base newTextureViewWithPixelFormat:view_format
                                       textureType:type
                                            levels:levels
                                            slices:slices];
    };
    const auto set = [&](TextureType type, MTLTextureType mtl_type, NSUInteger num_slices) {
        views[static_cast<size_t>(type)] = make(mtl_type, num_slices, sample_format, true);
    };
    const bool is_msaa = image.info.num_samples > 1;
    if (is_3d && info.type != ImageViewType::e3D) {
        // Vulkan can view 3D images as 2D arrays; Metal can't.
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "2D views of 3D images are not implemented");
        }
        return;
    }
    switch (info.type) {
    case ImageViewType::e1D:
    case ImageViewType::e1DArray:
    case ImageViewType::e2D:
    case ImageViewType::e2DArray:
    case ImageViewType::Rect: {
        const bool one_d =
            info.type == ImageViewType::e1D || info.type == ImageViewType::e1DArray;
        if (is_msaa) {
            set(TextureType::Color2D, MTLTextureType2DMultisample, 1);
            render_target = make(MTLTextureType2DMultisample, 1, host_format, false);
            break;
        }
        set(one_d ? TextureType::Color1D : TextureType::Color2D, MTLTextureType2D, 1);
        set(one_d ? TextureType::ColorArray1D : TextureType::ColorArray2D, MTLTextureType2DArray,
            all_slices.length);
        if (!one_d) {
            set(TextureType::Color2DRect, MTLTextureType2D, 1);
        }
        render_target = make(MTLTextureType2DArray, all_slices.length, host_format, false);
        break;
    }
    case ImageViewType::e3D:
        set(TextureType::Color3D, MTLTextureType3D, 1);
        render_target = make(MTLTextureType3D, 1, host_format, false);
        break;
    case ImageViewType::Cube:
    case ImageViewType::CubeArray:
        if (all_slices.length < 6 || is_msaa) {
            break;
        }
        set(TextureType::ColorCube, MTLTextureTypeCube, 6);
        if (all_slices.length % 6 == 0) {
            set(TextureType::ColorArrayCube, MTLTextureTypeCubeArray, all_slices.length);
        }
        render_target = make(MTLTextureType2DArray, all_slices.length, host_format, false);
        break;
    case ImageViewType::Buffer:
        ASSERT(false);
        break;
    }
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id_, Image& image, const SlotVector<Image>&)
    : ImageView{runtime, info, image_id_, image} {}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo& info,
                     const VideoCommon::ImageViewInfo& view_info, GPUVAddr gpu_addr_)
    : VideoCommon::ImageViewBase{info, view_info, gpu_addr_},
      buffer_size{VideoCommon::CalculateGuestSizeInBytes(info)} {}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::NullImageViewParams& params)
    : VideoCommon::ImageViewBase{params} {
    using Shader::TextureType;
    // Unbound textures read as zero, like Vulkan's null descriptors.
    id<MTLDevice> device = runtime.device.GetDevice();
    id<MTLTexture> array = NewNullTexture(device, MTLTextureType2DArray, 6);
    id<MTLTexture> volume = NewNullTexture(device, MTLTextureType3D, 1);
    null_textures = {array, volume};
    const MTLPixelFormat rgba8 = MTLPixelFormatRGBA8Unorm;
    const auto view = [&](MTLTextureType type, NSUInteger slices) {
        return [array newTextureViewWithPixelFormat:rgba8
                                        textureType:type
                                             levels:NSMakeRange(0, 1)
                                             slices:NSMakeRange(0, slices)];
    };
    views[static_cast<size_t>(TextureType::Color1D)] = view(MTLTextureType2D, 1);
    views[static_cast<size_t>(TextureType::ColorArray1D)] = view(MTLTextureType2DArray, 1);
    views[static_cast<size_t>(TextureType::Color2D)] = view(MTLTextureType2D, 1);
    views[static_cast<size_t>(TextureType::ColorArray2D)] = view(MTLTextureType2DArray, 1);
    views[static_cast<size_t>(TextureType::Color2DRect)] = view(MTLTextureType2D, 1);
    views[static_cast<size_t>(TextureType::ColorCube)] = view(MTLTextureTypeCube, 6);
    views[static_cast<size_t>(TextureType::ColorArrayCube)] = view(MTLTextureTypeCubeArray, 6);
    views[static_cast<size_t>(TextureType::Color3D)] = volume;
    host_format = rgba8;
}

ImageView::~ImageView() = default;

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc) {
    using namespace MaxwellToMTL::Sampler;
    const bool is_shadow_map = tsc.depth_compare_enabled != 0;
    if (tsc.reduction_filter != Tegra::Texture::SamplerReduction::WeightedAverage) {
        static bool logged{};
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Metal, "Min/max sampler reduction is not supported on Metal");
        }
    }
    // Some games have samplers with garbage. Sanitize them here.
    const float max_anisotropy = std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f);
    lod_bias = tsc.LodBias();

    const auto create = [&](float anisotropy) {
        MTLSamplerDescriptor* desc = [[MTLSamplerDescriptor alloc] init];
        desc.minFilter = Filter(tsc.min_filter);
        desc.magFilter = Filter(tsc.mag_filter);
        desc.mipFilter = MipFilter(tsc.mipmap_filter);
        desc.sAddressMode = WrapMode(tsc.wrap_u, tsc.mag_filter, is_shadow_map);
        desc.tAddressMode = WrapMode(tsc.wrap_v, tsc.mag_filter, is_shadow_map);
        desc.rAddressMode = WrapMode(tsc.wrap_p, tsc.mag_filter, is_shadow_map);
        // Shadow maps sample opaque white outside their bounds so nothing out there is shadowed.
        desc.borderColor = is_shadow_map ? MTLSamplerBorderColorOpaqueWhite
                                         : BorderColor(tsc.BorderColor());
        desc.lodMinClamp = tsc.MinLod();
        desc.lodMaxClamp = std::max(tsc.MinLod(), tsc.MaxLod());
        desc.maxAnisotropy = static_cast<NSUInteger>(anisotropy);
        if (is_shadow_map) {
            desc.compareFunction = DepthCompareFunction(tsc.depth_compare_func);
        }
        desc.normalizedCoordinates = YES;
        return [runtime.device.GetDevice() newSamplerStateWithDescriptor:desc];
    };
    sampler = create(max_anisotropy);
    const float max_anisotropy_default = static_cast<float>(1U << tsc.max_anisotropy);
    if (max_anisotropy > max_anisotropy_default) {
        sampler_default_anisotropy = create(max_anisotropy_default);
    }
}

Framebuffer::Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key)
    : width{key.size.width}, height{key.size.height} {
    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const color_buffer = color_buffers[index];
        if (!color_buffer) {
            continue;
        }
        color_attachments[index] = color_buffer->RenderTarget();
        color_formats[index] = color_buffer->HostFormat();
        width = std::min(width, color_buffer->size.width);
        height = std::min(height, color_buffer->size.height);
        num_layers = std::max(num_layers, static_cast<u32>(color_buffer->range.extent.layers));
        samples = color_buffer->Samples();
    }
    if (depth_buffer) {
        depth_attachment = depth_buffer->RenderTarget();
        depth_format = depth_buffer->HostFormat();
        width = std::min(width, depth_buffer->size.width);
        height = std::min(height, depth_buffer->size.height);
        num_layers = std::max(num_layers, static_cast<u32>(depth_buffer->range.extent.layers));
        samples = depth_buffer->Samples();
        const SurfaceType type = GetFormatType(depth_buffer->format);
        has_depth = type == SurfaceType::Depth || type == SurfaceType::DepthStencil;
        has_stencil = type == SurfaceType::Stencil || type == SurfaceType::DepthStencil;
    }
}

Framebuffer::~Framebuffer() = default;

MTLRenderPassDescriptor* Framebuffer::MakeRenderPassDescriptor() const {
    MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < NUM_RT; ++index) {
        if (color_attachments[index] == nil) {
            continue;
        }
        MTLRenderPassColorAttachmentDescriptor* attachment = desc.colorAttachments[index];
        attachment.texture = color_attachments[index];
        attachment.loadAction = MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionStore;
    }
    if (has_depth) {
        desc.depthAttachment.texture = depth_attachment;
        desc.depthAttachment.loadAction = MTLLoadActionLoad;
        desc.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (has_stencil) {
        desc.stencilAttachment.texture = depth_attachment;
        desc.stencilAttachment.loadAction = MTLLoadActionLoad;
        desc.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    if (num_layers > 1) {
        desc.renderTargetArrayLength = num_layers;
    }
    desc.renderTargetWidth = width;
    desc.renderTargetHeight = height;
    // Needed when the pass has no attachments.
    desc.defaultRasterSampleCount = samples;
    return desc;
}

} // namespace Metal
