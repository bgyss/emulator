// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <string>

#include "common/common_types.h"
#include "common/math_util.h"
#include "core/frontend/framebuffer_layout.h"

namespace Metal {

class Device;

/// Presents to a CAMetalLayer. Kept free of Objective-C types so it can
/// be used from plain C++; the implementation lives in mtl_presenter.mm.
class Presenter {
public:
    enum class LayerFormat {
        RGBA8, ///< Bytes in R, G, B, A order
        BGRA8, ///< Bytes in B, G, R, A order
    };

    /// How a layer is composited over the layers below it.
    enum class Blend {
        Opaque,
        Premultiplied,
        Coverage,
    };

    /// One guest framebuffer layer: linear pixels read from guest memory, or a texture the GPU
    /// rendered.
    struct Layer {
        u32 width{};
        u32 height{};
        LayerFormat format{};
        /// Tightly packed rows of width * 4 bytes. Ignored when texture is set.
        std::span<const u8> pixels;
        /// An id<MTLTexture> to sample instead of uploading pixels, bridged without a retain;
        /// the caller keeps it alive until Present returns. Present must be ordered after the
        /// work that renders it, on the same command queue.
        void* texture{};
        /// Normalized source rectangle; left > right or top > bottom flips the image.
        Common::Rectangle<f32> crop;
        Blend blending{};
    };

    /// @param layer CAMetalLayer to present to, or nullptr to run headless.
    /// @throws std::runtime_error when the present pipelines can't be created.
    explicit Presenter(const Device& device, void* layer);
    ~Presenter();

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    /// Clears the next drawable to the background color, draws the layers in order into the
    /// layout's screen rectangle, and presents it.
    void Present(std::span<const Layer> layers, const Layout::FramebufferLayout& layout, float red,
                 float green, float blue, bool vsync, bool linear_filter);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
