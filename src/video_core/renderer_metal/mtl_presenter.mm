// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <TargetConditionals.h>

#include <stdexcept>

#include "video_core/renderer_metal/mtl_presenter.h"

namespace Metal {

struct Presenter::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    CAMetalLayer* layer;
};

Presenter::Presenter(void* layer) : impl{std::make_unique<Impl>()} {
    impl->device = MTLCreateSystemDefaultDevice();
    if (impl->device == nil) {
        throw std::runtime_error("No Metal device available");
    }
    impl->queue = [impl->device newCommandQueue];
    if (impl->queue == nil) {
        throw std::runtime_error("Failed to create a Metal command queue");
    }

    impl->layer = (__bridge CAMetalLayer*)layer;
    if (impl->layer != nil) {
        impl->layer.device = impl->device;
        impl->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        impl->layer.framebufferOnly = YES;
    }
}

Presenter::~Presenter() = default;

void Presenter::Present(u32 width, u32 height, float red, float green, float blue, bool vsync) {
    if (impl->layer == nil || width == 0 || height == 0) {
        return;
    }
    @autoreleasepool {
        const CGSize size = CGSizeMake(width, height);
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

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(red, green, blue, 1.0);

        id<MTLCommandBuffer> command_buffer = [impl->queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder =
            [command_buffer renderCommandEncoderWithDescriptor:pass];
        [encoder endEncoding];
        [command_buffer presentDrawable:drawable];
        [command_buffer commit];
    }
}

std::string Presenter::GetDeviceName() const {
    return std::string{[[impl->device name] UTF8String]};
}

} // namespace Metal
