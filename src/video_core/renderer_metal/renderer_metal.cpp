// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/renderer_metal.h"

namespace Metal {

RendererMetal::RendererMetal(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), gpu(gpu_),
      presenter(emu_window.GetWindowInfo().render_surface), rasterizer(gpu_) {
    LOG_INFO(Render_Metal, "Metal device: {}", presenter.GetDeviceName());
    LOG_WARNING(Render_Metal, "The Metal renderer is experimental and does not draw guest "
                              "graphics yet");
}

RendererMetal::~RendererMetal() = default;

void RendererMetal::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (framebuffers.empty()) {
        return;
    }

    const auto& layout = render_window.GetFramebufferLayout();
    const bool vsync = Settings::values.vsync_mode.GetValue() != Settings::VSyncMode::Immediate;
    presenter.Present(layout.width, layout.height, Settings::values.bg_red.GetValue() / 255.0f,
                      Settings::values.bg_green.GetValue() / 255.0f,
                      Settings::values.bg_blue.GetValue() / 255.0f, vsync);

    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

std::vector<u8> RendererMetal::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

} // namespace Metal
