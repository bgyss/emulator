// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_clear_helper.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <unordered_map>

#include "common/common_types.h"

namespace Metal {

class Device;
class Scheduler;

/**
 * Clears render target attachments. A clear that covers a whole attachment uses a render pass
 * load action; a scissored or masked clear draws a triangle clipped to the scissor rectangle.
 */
class ClearHelper {
public:
    enum class ColorKind : u32 {
        Float,
        Uint,
        Sint,
    };

    explicit ClearHelper(const Device& device, Scheduler& scheduler);
    ~ClearHelper();

    ClearHelper(const ClearHelper&) = delete;
    ClearHelper& operator=(const ClearHelper&) = delete;

    /// Clears one slice of a color attachment.
    /// @param mask Components to write: bit 0 red, bit 1 green, bit 2 blue, bit 3 alpha.
    /// @param value Clear value; integer kinds hold exact integers.
    void ClearColor(id<MTLTexture> attachment, NSUInteger slice, MTLScissorRect rect, u8 mask,
                    ColorKind kind, const std::array<double, 4>& value);

    /// Clears one slice of a depth, stencil or depth-stencil attachment.
    void ClearDepthStencil(id<MTLTexture> attachment, NSUInteger slice, MTLScissorRect rect,
                           bool clear_depth, float depth, bool clear_stencil, u8 stencil,
                           u8 stencil_mask);

    [[nodiscard]] static bool HasDepth(MTLPixelFormat format);
    [[nodiscard]] static bool HasStencil(MTLPixelFormat format);

private:
    id<MTLRenderPipelineState> ColorPipeline(MTLPixelFormat format, NSUInteger samples,
                                             ColorKind kind, u8 mask);
    id<MTLRenderPipelineState> DepthStencilPipeline(MTLPixelFormat format, NSUInteger samples);
    id<MTLDepthStencilState> DepthStencilState(bool write_depth, bool write_stencil, u8 mask);
    id<MTLLibrary> Library();

    const Device& device;
    Scheduler& scheduler;
    id<MTLLibrary> library;
    std::unordered_map<u64, id<MTLRenderPipelineState>> pipelines;
    std::unordered_map<u32, id<MTLDepthStencilState>> depth_stencil_states;
};

} // namespace Metal
