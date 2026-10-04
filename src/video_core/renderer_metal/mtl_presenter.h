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

/// Owns the Metal device and presents to a CAMetalLayer. Kept free of Objective-C types so it can
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

    /// One guest framebuffer layer, already converted to linear pixels.
    struct Layer {
        u32 width{};
        u32 height{};
        LayerFormat format{};
        /// Tightly packed rows of width * 4 bytes.
        std::span<const u8> pixels;
        /// Normalized source rectangle; left > right or top > bottom flips the image.
        Common::Rectangle<f32> crop;
        Blend blending{};
    };

    /// @param layer CAMetalLayer to present to, or nullptr to run headless.
    /// @throws std::runtime_error when no Metal device is available or setup fails.
    explicit Presenter(void* layer);
    ~Presenter();

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    /// Clears the next drawable to the background color, draws the layers in order into the
    /// layout's screen rectangle, and presents it.
    void Present(std::span<const Layer> layers, const Layout::FramebufferLayout& layout, float red,
                 float green, float blue, bool vsync, bool linear_filter);

    [[nodiscard]] std::string GetDeviceName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
