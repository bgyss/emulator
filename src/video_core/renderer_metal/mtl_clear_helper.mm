// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "video_core/renderer_metal/mtl_clear_helper.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace Metal {

namespace {

constexpr const char* CLEAR_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct ClearParams {
    uint4 color; // bits of a float4, uint4 or int4
    float depth;
};

struct VertexOut {
    float4 position [[position]];
};

// One triangle covering the whole attachment; the scissor rectangle limits the clear.
vertex VertexOut clear_vertex(uint vertex_id [[vertex_id]],
                              constant ClearParams& p [[buffer(0)]]) {
    const float2 t = float2((vertex_id << 1) & 2, vertex_id & 2);
    VertexOut out;
    out.position = float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), p.depth, 1.0);
    return out;
}

fragment float4 clear_float(constant ClearParams& p [[buffer(0)]]) {
    return as_type<float4>(p.color);
}

fragment uint4 clear_uint(constant ClearParams& p [[buffer(0)]]) {
    return p.color;
}

fragment int4 clear_sint(constant ClearParams& p [[buffer(0)]]) {
    return as_type<int4>(p.color);
}
)";

struct ClearParams {
    std::array<u32, 4> color;
    float depth;
    std::array<u32, 3> padding;
};
static_assert(sizeof(ClearParams) == 32);

bool CoversWholeAttachment(id<MTLTexture> texture, MTLScissorRect rect) {
    return rect.x == 0 && rect.y == 0 && rect.width >= texture.width &&
           rect.height >= texture.height;
}

/// Points a render pass attachment at one slice of the texture.
void SetSlice(MTLRenderPassAttachmentDescriptor* attachment, id<MTLTexture> texture,
              NSUInteger slice) {
    attachment.texture = texture;
    switch (texture.textureType) {
    case MTLTextureType3D:
        attachment.depthPlane = slice;
        break;
    case MTLTextureType2DArray:
    case MTLTextureType2DMultisampleArray:
    case MTLTextureTypeCube:
    case MTLTextureTypeCubeArray:
        attachment.slice = slice;
        break;
    default:
        break;
    }
}

MTLScissorRect ClampRect(id<MTLTexture> texture, MTLScissorRect rect) {
    rect.x = std::min(rect.x, texture.width);
    rect.y = std::min(rect.y, texture.height);
    rect.width = std::min(rect.width, texture.width - rect.x);
    rect.height = std::min(rect.height, texture.height - rect.y);
    return rect;
}

MTLColorWriteMask WriteMask(u8 mask) {
    MTLColorWriteMask write_mask = MTLColorWriteMaskNone;
    if (mask & 1) {
        write_mask |= MTLColorWriteMaskRed;
    }
    if (mask & 2) {
        write_mask |= MTLColorWriteMaskGreen;
    }
    if (mask & 4) {
        write_mask |= MTLColorWriteMaskBlue;
    }
    if (mask & 8) {
        write_mask |= MTLColorWriteMaskAlpha;
    }
    return write_mask;
}

} // Anonymous namespace

ClearHelper::ClearHelper(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

ClearHelper::~ClearHelper() = default;

bool ClearHelper::HasDepth(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth16Unorm:
    case MTLPixelFormatDepth32Float:
    case MTLPixelFormatDepth32Float_Stencil8:
        return true;
    default:
        return false;
    }
}

bool ClearHelper::HasStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatStencil8 || format == MTLPixelFormatDepth32Float_Stencil8;
}

void ClearHelper::ClearColor(id<MTLTexture> attachment, NSUInteger slice, MTLScissorRect rect,
                             u8 mask, ColorKind kind, const std::array<double, 4>& value) {
    mask &= 0xF;
    if (attachment == nil || mask == 0) {
        return;
    }
    rect = ClampRect(attachment, rect);
    if (rect.width == 0 || rect.height == 0) {
        return;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    MTLRenderPassColorAttachmentDescriptor* color = pass.colorAttachments[0];
    SetSlice(color, attachment, slice);
    color.storeAction = MTLStoreActionStore;
    const bool full = mask == 0xF && CoversWholeAttachment(attachment, rect);
    if (full) {
        color.loadAction = MTLLoadActionClear;
        color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
    } else {
        color.loadAction = MTLLoadActionLoad;
    }
    scheduler.EndEncoding();
    id<MTLRenderCommandEncoder> encoder =
        [scheduler.CommandBuffer() renderCommandEncoderWithDescriptor:pass];
    if (!full) {
        ClearParams params{};
        for (size_t i = 0; i < 4; ++i) {
            switch (kind) {
            case ColorKind::Float: {
                const float component = static_cast<float>(value[i]);
                std::memcpy(&params.color[i], &component, sizeof(component));
                break;
            }
            case ColorKind::Uint:
                params.color[i] = static_cast<u32>(value[i]);
                break;
            case ColorKind::Sint:
                params.color[i] = static_cast<u32>(static_cast<s32>(value[i]));
                break;
            }
        }
        [encoder setRenderPipelineState:ColorPipeline(attachment.pixelFormat,
                                                      attachment.sampleCount, kind, mask)];
        [encoder setScissorRect:rect];
        [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
        [encoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [encoder endEncoding];
}

void ClearHelper::ClearDepthStencil(id<MTLTexture> attachment, NSUInteger slice,
                                    MTLScissorRect rect, bool clear_depth, float depth,
                                    bool clear_stencil, u8 stencil, u8 stencil_mask) {
    if (attachment == nil) {
        return;
    }
    const MTLPixelFormat format = attachment.pixelFormat;
    const bool has_depth = HasDepth(format);
    const bool has_stencil = HasStencil(format);
    clear_depth = clear_depth && has_depth;
    clear_stencil = clear_stencil && has_stencil && stencil_mask != 0;
    if (!clear_depth && !clear_stencil) {
        return;
    }
    rect = ClampRect(attachment, rect);
    if (rect.width == 0 || rect.height == 0) {
        return;
    }
    // A load-action clear writes every bit of the aspect, so a partial stencil mask needs a draw.
    const bool full = CoversWholeAttachment(attachment, rect) &&
                      (!clear_stencil || stencil_mask == 0xFF);

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (has_depth) {
        MTLRenderPassDepthAttachmentDescriptor* depth_attachment = pass.depthAttachment;
        SetSlice(depth_attachment, attachment, slice);
        depth_attachment.loadAction = full && clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
        depth_attachment.storeAction = MTLStoreActionStore;
        depth_attachment.clearDepth = depth;
    }
    if (has_stencil) {
        MTLRenderPassStencilAttachmentDescriptor* stencil_attachment = pass.stencilAttachment;
        SetSlice(stencil_attachment, attachment, slice);
        stencil_attachment.loadAction =
            full && clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
        stencil_attachment.storeAction = MTLStoreActionStore;
        stencil_attachment.clearStencil = stencil;
    }
    scheduler.EndEncoding();
    id<MTLRenderCommandEncoder> encoder =
        [scheduler.CommandBuffer() renderCommandEncoderWithDescriptor:pass];
    if (!full) {
        const ClearParams params{.color = {}, .depth = depth, .padding = {}};
        [encoder setRenderPipelineState:DepthStencilPipeline(format, attachment.sampleCount)];
        [encoder setDepthStencilState:DepthStencilState(clear_depth, clear_stencil,
                                                        stencil_mask)];
        [encoder setStencilReferenceValue:stencil];
        [encoder setScissorRect:rect];
        [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [encoder endEncoding];
}

id<MTLLibrary> ClearHelper::Library() {
    if (library == nil) {
        NSError* error = nil;
        library = [device.GetDevice() newLibraryWithSource:@(CLEAR_SHADER_SOURCE)
                                                   options:nil
                                                     error:&error];
        if (library == nil) {
            throw std::runtime_error(std::string{"Failed to compile Metal clear shaders: "} +
                                     error.localizedDescription.UTF8String);
        }
    }
    return library;
}

id<MTLRenderPipelineState> ClearHelper::ColorPipeline(MTLPixelFormat format, NSUInteger samples,
                                                      ColorKind kind, u8 mask) {
    const u64 key = static_cast<u64>(format) | (static_cast<u64>(kind) << 16) |
                    (static_cast<u64>(mask) << 20) | (static_cast<u64>(samples) << 24);
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }
    NSString* fragment = @"clear_float";
    if (kind == ColorKind::Uint) {
        fragment = @"clear_uint";
    } else if (kind == ColorKind::Sint) {
        fragment = @"clear_sint";
    }
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = [Library() newFunctionWithName:@"clear_vertex"];
    desc.fragmentFunction = [Library() newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat = format;
    desc.colorAttachments[0].writeMask = WriteMask(mask);
    desc.rasterSampleCount = samples;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline =
        [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string{"Failed to create Metal clear pipeline: "} +
                                 error.localizedDescription.UTF8String);
    }
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLRenderPipelineState> ClearHelper::DepthStencilPipeline(MTLPixelFormat format,
                                                             NSUInteger samples) {
    const u64 key = static_cast<u64>(format) | (static_cast<u64>(samples) << 24) | (1ULL << 40);
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = [Library() newFunctionWithName:@"clear_vertex"];
    // No fragment function: depth comes from the vertex position and stencil from the
    // reference value.
    if (HasDepth(format)) {
        desc.depthAttachmentPixelFormat = format;
    }
    if (HasStencil(format)) {
        desc.stencilAttachmentPixelFormat = format;
    }
    desc.rasterSampleCount = samples;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline =
        [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string{"Failed to create Metal depth clear pipeline: "} +
                                 error.localizedDescription.UTF8String);
    }
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLDepthStencilState> ClearHelper::DepthStencilState(bool write_depth, bool write_stencil,
                                                        u8 mask) {
    const u32 key = (write_depth ? 1U : 0U) | (write_stencil ? 2U : 0U) | (u32{mask} << 2);
    if (const auto it = depth_stencil_states.find(key); it != depth_stencil_states.end()) {
        return it->second;
    }
    MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
    desc.depthCompareFunction = MTLCompareFunctionAlways;
    desc.depthWriteEnabled = write_depth ? YES : NO;
    if (write_stencil) {
        MTLStencilDescriptor* stencil = [[MTLStencilDescriptor alloc] init];
        stencil.stencilCompareFunction = MTLCompareFunctionAlways;
        stencil.stencilFailureOperation = MTLStencilOperationReplace;
        stencil.depthFailureOperation = MTLStencilOperationReplace;
        stencil.depthStencilPassOperation = MTLStencilOperationReplace;
        stencil.writeMask = mask;
        desc.frontFaceStencil = stencil;
        desc.backFaceStencil = stencil;
    }
    id<MTLDepthStencilState> state = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
    depth_stencil_states.emplace(key, state);
    return state;
}

} // namespace Metal
