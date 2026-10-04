// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/renderer_metal/renderer_metal.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace Metal {

namespace {

/// Block height of guest framebuffers. Matches the Vulkan renderer's CPU path.
// TODO: Read this from HLE, like the Vulkan renderer's TODO says.
constexpr u32 FRAMEBUFFER_BLOCK_HEIGHT_LOG2 = 4;

u32 GetBytesPerPixel(const Tegra::FramebufferConfig& framebuffer) {
    using namespace VideoCore::Surface;
    return BytesPerBlock(PixelFormatFromGPUPixelFormat(framebuffer.pixel_format));
}

Presenter::Blend ToPresenterBlend(Tegra::BlendMode mode) {
    switch (mode) {
    case Tegra::BlendMode::Premultiplied:
        return Presenter::Blend::Premultiplied;
    case Tegra::BlendMode::Coverage:
        return Presenter::Blend::Coverage;
    case Tegra::BlendMode::Opaque:
    default:
        return Presenter::Blend::Opaque;
    }
}

/// Expands R5G6B5 (red in the high bits) to RGBA8.
void ExpandRgb565(std::span<const u8> source, std::span<u8> dest, size_t pixel_count) {
    for (size_t i = 0; i < pixel_count; ++i) {
        const u16 value = static_cast<u16>(source[i * 2] | (source[i * 2 + 1] << 8));
        const u32 r = (value >> 11) & 0x1F;
        const u32 g = (value >> 5) & 0x3F;
        const u32 b = value & 0x1F;
        dest[i * 4 + 0] = static_cast<u8>((r << 3) | (r >> 2));
        dest[i * 4 + 1] = static_cast<u8>((g << 2) | (g >> 4));
        dest[i * 4 + 2] = static_cast<u8>((b << 3) | (b >> 2));
        dest[i * 4 + 3] = 0xFF;
    }
}

} // Anonymous namespace

RendererMetal::RendererMetal(Core::Frontend::EmuWindow& emu_window,
                             Tegra::MaxwellDeviceMemoryManager& device_memory_, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), device_memory(device_memory_), gpu(gpu_),
      device(std::make_unique<Device>()), scheduler(std::make_unique<Scheduler>(*device)),
      staging_buffer_pool(std::make_unique<StagingBufferPool>(*device, *scheduler)),
      presenter(*device, emu_window.GetWindowInfo().render_surface),
      rasterizer(std::make_unique<RasterizerMetal>(gpu_, device_memory_, *device, *scheduler,
                                                   *staging_buffer_pool)) {
    LOG_WARNING(Render_Metal, "The Metal renderer is experimental: it draws with vertex and "
                              "fragment shaders, but compute, tessellation, geometry shaders and "
                              "texture buffers are not implemented yet");
}

RendererMetal::~RendererMetal() = default;

VideoCore::RasterizerInterface* RendererMetal::ReadRasterizer() {
    return rasterizer.get();
}

std::string RendererMetal::GetDeviceVendor() const {
    return device->GetName();
}

bool RendererMetal::ReadFramebuffer(const Tegra::FramebufferConfig& framebuffer,
                                    std::vector<u8>& pixels, Presenter::Layer& layer) {
    if (framebuffer.width == 0 || framebuffer.height == 0) {
        return false;
    }
    const DAddr address = framebuffer.address + framebuffer.offset;
    const u8* const host_ptr = device_memory.GetPointer<u8>(address);
    if (host_ptr == nullptr) {
        return false;
    }

    const u32 bytes_per_pixel = GetBytesPerPixel(framebuffer);
    const size_t tiled_size =
        Tegra::Texture::CalculateSize(true, bytes_per_pixel, framebuffer.stride, framebuffer.height,
                                      1, FRAMEBUFFER_BLOCK_HEIGHT_LOG2, 0);
    const size_t linear_size = size_t{framebuffer.stride} * framebuffer.height * bytes_per_pixel;
    const size_t pixel_count = size_t{framebuffer.width} * framebuffer.height;

    const bool is_rgb565 = framebuffer.pixel_format == Service::android::PixelFormat::Rgb565;
    std::vector<u8>& linear = is_rgb565 ? rgb565_scratch : pixels;
    linear.resize(std::max(linear_size, pixel_count * bytes_per_pixel));
    Tegra::Texture::UnswizzleTexture(linear, std::span(host_ptr, tiled_size), bytes_per_pixel,
                                     framebuffer.width, framebuffer.height, 1,
                                     FRAMEBUFFER_BLOCK_HEIGHT_LOG2, 0);

    switch (framebuffer.pixel_format) {
    case Service::android::PixelFormat::Rgb565:
        pixels.resize(pixel_count * 4);
        ExpandRgb565(linear, pixels, pixel_count);
        layer.format = Presenter::LayerFormat::RGBA8;
        break;
    case Service::android::PixelFormat::Rgbx8888:
        for (size_t i = 0; i < pixel_count; ++i) {
            pixels[i * 4 + 3] = 0xFF;
        }
        layer.format = Presenter::LayerFormat::RGBA8;
        break;
    case Service::android::PixelFormat::Bgra8888:
        layer.format = Presenter::LayerFormat::BGRA8;
        break;
    case Service::android::PixelFormat::Rgba8888:
        layer.format = Presenter::LayerFormat::RGBA8;
        break;
    default:
        LOG_ERROR(Render_Metal, "Unsupported framebuffer pixel format: {}",
                  static_cast<u32>(framebuffer.pixel_format));
        return false;
    }

    layer.width = framebuffer.width;
    layer.height = framebuffer.height;
    layer.pixels = std::span<const u8>(pixels.data(), pixel_count * 4);
    layer.crop = Tegra::NormalizeCrop(framebuffer, framebuffer.width, framebuffer.height);
    layer.blending = ToPresenterBlend(framebuffer.blending);
    return true;
}

void RendererMetal::LogLayerChange(size_t index, const Tegra::FramebufferConfig& framebuffer,
                                   bool readable, bool rendered) {
    if (logged_layers.size() <= index) {
        logged_layers.resize(index + 1);
    }
    // The offset is left out: games flip between buffers in the same allocation every frame.
    std::string description = fmt::format(
        "{}x{} stride {} format {} blending {} address {:#x}{}", framebuffer.width,
        framebuffer.height, framebuffer.stride, static_cast<u32>(framebuffer.pixel_format),
        static_cast<u32>(framebuffer.blending), framebuffer.address,
        rendered ? " (GPU-rendered)" : (readable ? "" : " (not readable)"));
    if (logged_layers[index] != description) {
        LOG_WARNING(Render_Metal, "Layer {}: {}", index, description);
        logged_layers[index] = std::move(description);
    }
}

void RendererMetal::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (framebuffers.empty()) {
        return;
    }

    if (layer_pixels.size() < framebuffers.size()) {
        layer_pixels.resize(framebuffers.size());
    }
    std::vector<Presenter::Layer> layers;
    layers.reserve(framebuffers.size());
    // Keeps the rendered textures alive until Present returns.
    std::vector<id<MTLTexture>> rendered_textures;
    for (size_t i = 0; i < framebuffers.size(); ++i) {
        const Tegra::FramebufferConfig& framebuffer = framebuffers[i];
        Presenter::Layer layer;
        const DAddr address = framebuffer.address + framebuffer.offset;
        if (const auto rendered = rasterizer->AccelerateDisplay(framebuffer, address)) {
            // The GPU rendered this framebuffer; present its texture.
            layer.width = rendered->width;
            layer.height = rendered->height;
            layer.texture = (__bridge void*)rendered->texture;
            layer.crop = Tegra::NormalizeCrop(framebuffer, rendered->width, rendered->height);
            layer.blending = ToPresenterBlend(framebuffer.blending);
            rendered_textures.push_back(rendered->texture);
            LogLayerChange(i, framebuffer, true, true);
            layers.push_back(layer);
            continue;
        }
        const bool readable = ReadFramebuffer(framebuffer, layer_pixels[i], layer);
        LogLayerChange(i, framebuffer, readable, false);
        if (readable) {
            layers.push_back(layer);
        }
    }

    // Submit this frame's rendering and transfers ahead of the present, which uses the same
    // queue.
    scheduler->Flush();

    const auto& layout = render_window.GetFramebufferLayout();
    const bool vsync = Settings::values.vsync_mode.GetValue() != Settings::VSyncMode::Immediate;
    const bool linear_filter =
        Settings::values.scaling_filter.GetValue() != Settings::ScalingFilter::NearestNeighbor;
    presenter.Present(layers, layout, Settings::values.bg_red.GetValue() / 255.0f,
                      Settings::values.bg_green.GetValue() / 255.0f,
                      Settings::values.bg_blue.GetValue() / 255.0f, vsync, linear_filter);
    rasterizer->TickFrame();

    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

std::vector<u8> RendererMetal::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

} // namespace Metal
