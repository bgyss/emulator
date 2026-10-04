// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <TargetConditionals.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_presenter.h"

namespace Metal {

namespace {

constexpr size_t FRAMES_IN_FLIGHT = 3;

constexpr const char* PRESENT_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct ScreenVertex {
    float2 position;
    float2 uv;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut present_vertex(uint vertex_id [[vertex_id]],
                                constant ScreenVertex* vertices [[buffer(0)]]) {
    VertexOut out;
    out.position = float4(vertices[vertex_id].position, 0.0, 1.0);
    out.uv = vertices[vertex_id].uv;
    return out;
}

fragment float4 present_fragment(VertexOut in [[stage_in]],
                                 texture2d<float> source [[texture(0)]],
                                 sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
)";

struct ScreenVertex {
    float x, y;
    float u, v;
};

size_t BlendIndex(Presenter::Blend mode) {
    switch (mode) {
    case Presenter::Blend::Premultiplied:
        return 1;
    case Presenter::Blend::Coverage:
        return 2;
    case Presenter::Blend::Opaque:
    default:
        return 0;
    }
}

} // Anonymous namespace

struct Presenter::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    CAMetalLayer* layer;
    /// Indexed by BlendIndex.
    std::array<id<MTLRenderPipelineState>, 3> pipelines;
    id<MTLSamplerState> linear_sampler;
    id<MTLSamplerState> nearest_sampler;
    /// Limits the CPU to FRAMES_IN_FLIGHT frames ahead so it never rewrites a texture the GPU is
    /// still sampling.
    dispatch_semaphore_t frame_semaphore;
    /// Layer textures for each frame in flight.
    std::array<std::vector<id<MTLTexture>>, FRAMES_IN_FLIGHT> layer_textures;
    size_t frame_index{};

    id<MTLRenderPipelineState> CreatePipeline(id<MTLLibrary> library, Presenter::Blend mode) {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = [library newFunctionWithName:@"present_vertex"];
        desc.fragmentFunction = [library newFunctionWithName:@"present_fragment"];
        MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[0];
        color.pixelFormat = MTLPixelFormatBGRA8Unorm;
        if (mode != Presenter::Blend::Opaque) {
            color.blendingEnabled = YES;
            color.sourceRGBBlendFactor = mode == Presenter::Blend::Premultiplied
                                             ? MTLBlendFactorOne
                                             : MTLBlendFactorSourceAlpha;
            color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            color.sourceAlphaBlendFactor = MTLBlendFactorOne;
            color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
        NSError* error = nil;
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:desc
                                                                                     error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(std::string{"Failed to create Metal present pipeline: "} +
                                     error.localizedDescription.UTF8String);
        }
        return pipeline;
    }

    id<MTLSamplerState> CreateSampler(MTLSamplerMinMagFilter filter) {
        MTLSamplerDescriptor* desc = [[MTLSamplerDescriptor alloc] init];
        desc.minFilter = filter;
        desc.magFilter = filter;
        desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        return [device newSamplerStateWithDescriptor:desc];
    }

    id<MTLTexture> UploadLayer(size_t index, const Presenter::Layer& source) {
        auto& textures = layer_textures[frame_index];
        if (textures.size() <= index) {
            textures.resize(index + 1);
        }
        const MTLPixelFormat format = source.format == Presenter::LayerFormat::BGRA8
                                          ? MTLPixelFormatBGRA8Unorm
                                          : MTLPixelFormatRGBA8Unorm;
        id<MTLTexture> texture = textures[index];
        if (texture == nil || texture.width != source.width || texture.height != source.height ||
            texture.pixelFormat != format) {
            MTLTextureDescriptor* desc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                   width:source.width
                                                                  height:source.height
                                                               mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModeShared;
            texture = [device newTextureWithDescriptor:desc];
            textures[index] = texture;
        }
        [texture replaceRegion:MTLRegionMake2D(0, 0, source.width, source.height)
                   mipmapLevel:0
                     withBytes:source.pixels.data()
                   bytesPerRow:source.width * 4];
        return texture;
    }
};

Presenter::Presenter(const Device& device, void* layer) : impl{std::make_unique<Impl>()} {
    impl->device = device.GetDevice();
    impl->queue = device.GetQueue();

    NSError* error = nil;
    id<MTLLibrary> library =
        [impl->device newLibraryWithSource:@(PRESENT_SHADER_SOURCE) options:nil error:&error];
    if (library == nil) {
        throw std::runtime_error(std::string{"Failed to compile Metal present shaders: "} +
                                 error.localizedDescription.UTF8String);
    }
    impl->pipelines[BlendIndex(Presenter::Blend::Opaque)] =
        impl->CreatePipeline(library, Presenter::Blend::Opaque);
    impl->pipelines[BlendIndex(Presenter::Blend::Premultiplied)] =
        impl->CreatePipeline(library, Presenter::Blend::Premultiplied);
    impl->pipelines[BlendIndex(Presenter::Blend::Coverage)] =
        impl->CreatePipeline(library, Presenter::Blend::Coverage);
    impl->linear_sampler = impl->CreateSampler(MTLSamplerMinMagFilterLinear);
    impl->nearest_sampler = impl->CreateSampler(MTLSamplerMinMagFilterNearest);
    impl->frame_semaphore = dispatch_semaphore_create(FRAMES_IN_FLIGHT);

    impl->layer = (__bridge CAMetalLayer*)layer;
    if (impl->layer != nil) {
        impl->layer.device = impl->device;
        impl->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        impl->layer.framebufferOnly = YES;
    }
}

Presenter::~Presenter() {
    // Let in-flight frames finish before their textures are released.
    for (size_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        dispatch_semaphore_wait(impl->frame_semaphore, DISPATCH_TIME_FOREVER);
    }
    for (size_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        dispatch_semaphore_signal(impl->frame_semaphore);
    }
}

void Presenter::Present(std::span<const Layer> layers, const Layout::FramebufferLayout& layout,
                        float red, float green, float blue, bool vsync, bool linear_filter) {
    if (impl->layer == nil || layout.width == 0 || layout.height == 0) {
        return;
    }
    @autoreleasepool {
        const CGSize size = CGSizeMake(layout.width, layout.height);
        if (!CGSizeEqualToSize(impl->layer.drawableSize, size)) {
            impl->layer.drawableSize = size;
        }
#if TARGET_OS_OSX
        impl->layer.displaySyncEnabled = vsync ? YES : NO;
#else
        (void)vsync;
#endif
        id<CAMetalDrawable> drawable = [impl->layer nextDrawable];
        if (drawable == nil) {
            return;
        }

        dispatch_semaphore_wait(impl->frame_semaphore, DISPATCH_TIME_FOREVER);
        impl->frame_index = (impl->frame_index + 1) % FRAMES_IN_FLIGHT;

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(red, green, blue, 1.0);

        id<MTLCommandBuffer> command_buffer = [impl->queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder =
            [command_buffer renderCommandEncoderWithDescriptor:pass];

        // Map the layout's screen rectangle from window pixels to normalized device coordinates.
        const float width = static_cast<float>(layout.width);
        const float height = static_cast<float>(layout.height);
        const float left = static_cast<float>(layout.screen.left) / width * 2.0f - 1.0f;
        const float right = static_cast<float>(layout.screen.right) / width * 2.0f - 1.0f;
        const float top = 1.0f - static_cast<float>(layout.screen.top) / height * 2.0f;
        const float bottom = 1.0f - static_cast<float>(layout.screen.bottom) / height * 2.0f;

        [encoder setFragmentSamplerState:linear_filter ? impl->linear_sampler
                                                       : impl->nearest_sampler
                                 atIndex:0];
        for (size_t i = 0; i < layers.size(); ++i) {
            const Layer& source = layers[i];
            id<MTLTexture> rendered = (__bridge id<MTLTexture>)source.texture;
            if (source.width == 0 || source.height == 0 ||
                (rendered == nil &&
                 source.pixels.size() < size_t{source.width} * source.height * 4)) {
                continue;
            }
            const std::array<ScreenVertex, 4> vertices{{
                {left, top, source.crop.left, source.crop.top},
                {right, top, source.crop.right, source.crop.top},
                {left, bottom, source.crop.left, source.crop.bottom},
                {right, bottom, source.crop.right, source.crop.bottom},
            }};
            [encoder setRenderPipelineState:impl->pipelines[BlendIndex(source.blending)]];
            [encoder setVertexBytes:vertices.data() length:sizeof(vertices) atIndex:0];
            [encoder setFragmentTexture:rendered != nil ? rendered : impl->UploadLayer(i, source)
                                atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
        [encoder endEncoding];

        dispatch_semaphore_t semaphore = impl->frame_semaphore;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          dispatch_semaphore_signal(semaphore);
        }];
        [command_buffer presentDrawable:drawable];
        [command_buffer commit];
    }
}

} // namespace Metal
