// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_texture_cache.h holds Objective-C types; include it from Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <span>
#include <unordered_map>
#include <utility>

#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/texture_cache_base.h"

namespace Metal {

using Common::SlotVector;
using VideoCommon::ImageId;
using VideoCommon::NUM_RT;
using VideoCommon::Region2D;
using VideoCommon::RenderTargets;
using VideoCore::Surface::PixelFormat;

class Device;
class Framebuffer;
class Image;
class ImageView;
class Scheduler;

/// The texture behind an image or image view, for texel copies.
struct TextureCopyTarget {
    id<MTLTexture> texture;
    /// Guest format with the texture's data layout, see MaxwellToMTL::HostDataFormat.
    PixelFormat data_format{PixelFormat::Invalid};
    bool is_3d{};
};

/**
 * Backend half of VideoCommon::TextureCache for Metal.
 *
 * Images are private-storage MTLTextures. Uploads, downloads and copies go through the
 * scheduler's blit encoder; copies between different formats of the same block size go through
 * a staging buffer, since Metal blits only copy between identical formats. Scaled color blits
 * draw a quad. Metal tracks hazards between encoders for these textures, so there are no
 * barriers or layout transitions.
 */
class TextureCacheRuntime {
public:
    explicit TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                 StagingBufferPool& staging_buffer_pool_);
    ~TextureCacheRuntime();

    TextureCacheRuntime(const TextureCacheRuntime&) = delete;
    TextureCacheRuntime& operator=(const TextureCacheRuntime&) = delete;

    void Finish();

    StagingBufferRef UploadStagingBuffer(size_t size);

    StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void TickFrame() {}

    u64 GetDeviceLocalMemory() const;

    u64 GetDeviceMemoryUsage() const;

    bool CanReportMemoryUsage() const {
        return true;
    }

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Region2D& dst_region, const Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    bool ShouldReinterpret(Image& dst, Image& src);

    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);

    bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }

    bool CanUploadMSAA() const noexcept {
        return false;
    }

    void AccelerateImageUpload(Image&, const StagingBufferRef&,
                               std::span<const VideoCommon::SwizzleParameters>) {}

    void InsertUploadMemoryBarrier() {}

    void TransitionImageLayout(Image&) {}

    bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }

    bool HasNativeBgr() const noexcept {
        return true;
    }

    void BarrierFeedbackLoop() {}

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;

private:
    /// Copies texels with the blit encoder; both textures must have the same format.
    void BlitCopy(const TextureCopyTarget& dst, const TextureCopyTarget& src,
                  std::span<const VideoCommon::ImageCopy> copies);

    /// Copies texels through a staging buffer, for formats a blit can't convert between. Both
    /// data formats must have the same bytes per block.
    void CopyThroughBuffer(const TextureCopyTarget& dst, const TextureCopyTarget& src,
                           std::span<const VideoCommon::ImageCopy> copies);

    /// Draws src_region of src into dst_region of dst with filtering.
    void DrawBlit(ImageView& dst, ImageView& src, const Region2D& dst_region,
                  const Region2D& src_region, bool linear);

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format, u32 kind);

    id<MTLLibrary> blit_library;
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;
    id<MTLSamplerState> nearest_sampler;
    id<MTLSamplerState> linear_sampler;
};

class Image : public VideoCommon::ImageBase {
public:
    explicit Image(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info,
                   GPUVAddr gpu_addr, VAddr cpu_addr);
    explicit Image(const VideoCommon::NullImageParams&);

    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void UploadMemory(id<MTLBuffer> buffer, size_t offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void UploadMemory(const StagingBufferRef& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(id<MTLBuffer> buffer, size_t offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(std::span<id<MTLBuffer>> buffers, std::span<size_t> offsets,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(const StagingBufferRef& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    [[nodiscard]] id<MTLTexture> Handle() const noexcept {
        return texture;
    }

    /// Host format of the texture.
    [[nodiscard]] MTLPixelFormat HostFormat() const noexcept {
        return host_format;
    }

    /// Guest format with the memory layout of the texture's data, see MaxwellToMTL.
    [[nodiscard]] PixelFormat DataFormat() const noexcept {
        return data_format;
    }

    [[nodiscard]] TextureCopyTarget CopyTarget() const noexcept {
        return {
            .texture = texture,
            .data_format = data_format,
            .is_3d = info.type == VideoCommon::ImageType::e3D,
        };
    }

    /// Returns true when the image is already initialized and mark it as initialized
    [[nodiscard]] bool ExchangeInitialization() noexcept {
        return std::exchange(initialized, true);
    }

    /// Resolution scaling is not implemented on Metal yet; images are never rescaled.
    bool IsRescaled() const noexcept {
        return false;
    }

    bool ScaleUp(bool = false) {
        return false;
    }

    bool ScaleDown(bool = false) {
        return false;
    }

private:
    TextureCacheRuntime* runtime{};
    id<MTLTexture> texture;
    MTLPixelFormat host_format{MTLPixelFormatInvalid};
    PixelFormat data_format{PixelFormat::Invalid};
    bool initialized{};
};

class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&,
                       const SlotVector<Image>&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo&,
                       const VideoCommon::ImageViewInfo&, GPUVAddr);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams&);

    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// The view to sample as the given shader texture type; nil when the view can't be one.
    [[nodiscard]] id<MTLTexture> Handle(Shader::TextureType texture_type) const noexcept {
        return views[static_cast<size_t>(texture_type)];
    }

    /// The whole range of the view in its own format with no swizzle, for render pass
    /// attachments.
    [[nodiscard]] id<MTLTexture> RenderTarget() const noexcept {
        return render_target;
    }

    [[nodiscard]] id<MTLTexture> ImageHandle() const noexcept {
        return copy_target.texture;
    }

    /// The image's texture, for copies of the view's subresources.
    [[nodiscard]] const TextureCopyTarget& CopyTarget() const noexcept {
        return copy_target;
    }

    [[nodiscard]] MTLPixelFormat HostFormat() const noexcept {
        return host_format;
    }

    [[nodiscard]] NSUInteger Samples() const noexcept {
        return samples;
    }

    [[nodiscard]] GPUVAddr GpuAddr() const noexcept {
        return gpu_addr;
    }

    [[nodiscard]] u32 BufferSize() const noexcept {
        return buffer_size;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

private:
    std::array<id<MTLTexture>, Shader::NUM_TEXTURE_TYPES> views;
    id<MTLTexture> render_target;
    /// Null views own their textures; other views reference the image's.
    std::array<id<MTLTexture>, 2> null_textures;
    TextureCopyTarget copy_target;
    MTLPixelFormat host_format{MTLPixelFormatInvalid};
    NSUInteger samples{1};
    u32 buffer_size{};
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    [[nodiscard]] id<MTLSamplerState> Handle() const noexcept {
        return sampler;
    }

    [[nodiscard]] id<MTLSamplerState> HandleWithDefaultAnisotropy() const noexcept {
        return sampler_default_anisotropy != nil ? sampler_default_anisotropy : sampler;
    }

    [[nodiscard]] bool HasAddedAnisotropy() const noexcept {
        return sampler_default_anisotropy != nil;
    }

    /// Metal samplers have no LOD bias; shaders have to apply it.
    [[nodiscard]] float LodBias() const noexcept {
        return lod_bias;
    }

private:
    id<MTLSamplerState> sampler;
    id<MTLSamplerState> sampler_default_anisotropy;
    float lod_bias{};
};

/// The attachments of a render pass. Metal has no framebuffer objects; the rasterizer builds an
/// MTLRenderPassDescriptor from this when it starts a pass.
class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime& runtime, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key);

    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&&) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    /// A render pass descriptor that loads and stores every attachment.
    [[nodiscard]] MTLRenderPassDescriptor* MakeRenderPassDescriptor() const;

    [[nodiscard]] id<MTLTexture> ColorAttachment(size_t index) const noexcept {
        return color_attachments[index];
    }

    [[nodiscard]] MTLPixelFormat ColorFormat(size_t index) const noexcept {
        return color_formats[index];
    }

    [[nodiscard]] id<MTLTexture> DepthAttachment() const noexcept {
        return depth_attachment;
    }

    [[nodiscard]] MTLPixelFormat DepthFormat() const noexcept {
        return depth_format;
    }

    [[nodiscard]] u32 Width() const noexcept {
        return width;
    }

    [[nodiscard]] u32 Height() const noexcept {
        return height;
    }

    [[nodiscard]] NSUInteger Samples() const noexcept {
        return samples;
    }

    [[nodiscard]] u32 NumLayers() const noexcept {
        return num_layers;
    }

    [[nodiscard]] bool HasAspectDepthBit() const noexcept {
        return has_depth;
    }

    [[nodiscard]] bool HasAspectStencilBit() const noexcept {
        return has_stencil;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

private:
    std::array<id<MTLTexture>, NUM_RT> color_attachments;
    std::array<MTLPixelFormat, NUM_RT> color_formats{};
    id<MTLTexture> depth_attachment;
    MTLPixelFormat depth_format{MTLPixelFormatInvalid};
    u32 width{};
    u32 height{};
    u32 num_layers{1};
    NSUInteger samples{1};
    bool has_depth{};
    bool has_stencil{};
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = true;

    using Runtime = Metal::TextureCacheRuntime;
    using Image = Metal::Image;
    using ImageAlloc = Metal::ImageAlloc;
    using ImageView = Metal::ImageView;
    using Sampler = Metal::Sampler;
    using Framebuffer = Metal::Framebuffer;
    using AsyncBuffer = Metal::StagingBufferRef;
    using BufferType = id<MTLBuffer>;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace Metal
