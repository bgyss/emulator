// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>

#include "video_core/renderer_base.h"
#include "video_core/renderer_metal/mtl_presenter.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace Metal {

/// Experimental native Metal renderer. For now it only presents a cleared frame to the window;
/// guest rendering goes through the null rasterizer until the Metal rasterizer exists.
class RendererMetal final : public VideoCore::RendererBase {
public:
    explicit RendererMetal(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu,
                           std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererMetal() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override {
        return presenter.GetDeviceName();
    }

private:
    Tegra::GPU& gpu;
    Presenter presenter;
    Null::RasterizerNull rasterizer;
};

} // namespace Metal
