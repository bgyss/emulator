// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_metal/mtl_presenter.h"

namespace Metal {

class Device;
class RasterizerMetal;
class Scheduler;
class StagingBufferPool;

/// Experimental native Metal renderer. It presents framebuffers the GPU cleared or copied into
/// straight from the texture cache, and reads the rest from guest memory. Draws are not
/// implemented yet, so most GPU-rendered graphics don't show up.
class RendererMetal final : public VideoCore::RendererBase {
public:
    explicit RendererMetal(Core::Frontend::EmuWindow& emu_window,
                           Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu,
                           std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererMetal() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override;

    [[nodiscard]] std::string GetDeviceVendor() const override;

private:
    /// Reads a guest framebuffer into linear pixels. Returns false when it can't be read.
    bool ReadFramebuffer(const Tegra::FramebufferConfig& framebuffer, std::vector<u8>& pixels,
                         Presenter::Layer& layer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Tegra::GPU& gpu;
    std::unique_ptr<Device> device;
    std::unique_ptr<Scheduler> scheduler;
    std::unique_ptr<StagingBufferPool> staging_buffer_pool;
    Presenter presenter;
    std::unique_ptr<RasterizerMetal> rasterizer;

    /// Logs a layer's configuration when it differs from the previous frame's.
    void LogLayerChange(size_t index, const Tegra::FramebufferConfig& framebuffer, bool readable,
                        bool rendered);

    /// Last logged configuration of each layer, to log only when it changes.
    std::vector<std::string> logged_layers;
    /// Linear RGBA8/BGRA8 pixels of each layer, reused across frames.
    std::vector<std::vector<u8>> layer_pixels;
    /// Unswizzled R5G6B5 pixels before expansion to RGBA8.
    std::vector<u8> rgb565_scratch;
};

} // namespace Metal
