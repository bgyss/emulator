// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_staging_buffer_pool.h"

using namespace Metal;
using VideoCommon::BufferCopy;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

namespace {

std::unique_ptr<Device> MakeDevice() {
    try {
        return std::make_unique<Device>();
    } catch (const std::runtime_error& error) {
        WARN("No Metal device; skipping: " << error.what());
        return nullptr;
    }
}

struct Fixture {
    explicit Fixture(const Device& device)
        : scheduler{device}, pool{device, scheduler}, runtime{device, scheduler, pool} {}

    void Upload(id<MTLBuffer> dst, size_t dst_offset, std::span<const u8> data) {
        const StagingBufferRef staging = runtime.UploadStagingBuffer(data.size());
        std::memcpy(staging.mapped_span.data(), data.data(), data.size());
        const std::array copies{BufferCopy{
            .src_offset = staging.offset,
            .dst_offset = dst_offset,
            .size = data.size(),
        }};
        runtime.CopyBuffer(dst, staging.buffer, copies, true);
    }

    std::vector<u8> Download(id<MTLBuffer> src, size_t src_offset, size_t size) {
        const StagingBufferRef staging = runtime.DownloadStagingBuffer(size);
        const std::array copies{BufferCopy{
            .src_offset = src_offset,
            .dst_offset = staging.offset,
            .size = size,
        }};
        runtime.CopyBuffer(staging.buffer, src, copies, true);
        runtime.Finish();
        const u8* const data = staging.mapped_span.data();
        return std::vector<u8>(data, data + size);
    }

    /// Reads the indices of the current index binding; waits for the GPU first.
    template <typename T>
    std::vector<T> ReadIndices(size_t count) {
        runtime.Finish();
        const IndexBufferBinding& binding = runtime.GetIndexBuffer();
        const u8* const data = static_cast<const u8*>(binding.buffer.contents) + binding.offset;
        std::vector<T> indices(count);
        std::memcpy(indices.data(), data, count * sizeof(T));
        return indices;
    }

    Scheduler scheduler;
    StagingBufferPool pool;
    BufferCacheRuntime runtime;
};

std::vector<u8> Iota(size_t size, u8 first = 0) {
    std::vector<u8> data(size);
    std::iota(data.begin(), data.end(), first);
    return data;
}

} // Anonymous namespace

TEST_CASE("Metal buffer cache round-trips data through a buffer", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);
    REQUIRE(buffer.Handle() != nil);
    REQUIRE(buffer.Handle().length >= 0x10000);
    REQUIRE(buffer.Handle().storageMode == MTLStorageModePrivate);

    const std::vector<u8> data = Iota(4096, 7);
    fixture.Upload(buffer, 256, data);
    REQUIRE(fixture.Download(buffer, 256, data.size()) == data);
}

TEST_CASE("Metal buffer cache copies between buffers", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer src(fixture.runtime, 0x10000, 0x10000);
    Buffer dst(fixture.runtime, 0x20000, 0x10000);

    const std::vector<u8> data = Iota(1024);
    fixture.Upload(src, 0, data);
    fixture.runtime.ClearBuffer(dst, 0, 1024, 0);
    const std::array copies{
        BufferCopy{.src_offset = 0, .dst_offset = 512, .size = 256},
        BufferCopy{.src_offset = 512, .dst_offset = 0, .size = 256},
    };
    fixture.runtime.CopyBuffer(dst, src, copies, true);

    const std::vector<u8> result = fixture.Download(dst, 0, 1024);
    REQUIRE(std::equal(result.begin(), result.begin() + 256, data.begin() + 512));
    REQUIRE(std::all_of(result.begin() + 256, result.begin() + 512, [](u8 v) { return v == 0; }));
    REQUIRE(std::equal(result.begin() + 512, result.begin() + 768, data.begin()));
}

TEST_CASE("Metal buffer cache clears with byte and word patterns", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);

    fixture.runtime.ClearBuffer(buffer, 0, 1024, 0xABABABABU);
    fixture.runtime.ClearBuffer(buffer, 256, 512, 0x12345678U);
    const std::vector<u8> result = fixture.Download(buffer, 0, 1024);

    REQUIRE(std::all_of(result.begin(), result.begin() + 256, [](u8 v) { return v == 0xAB; }));
    for (size_t i = 256; i < 768; i += sizeof(u32)) {
        u32 word;
        std::memcpy(&word, result.data() + i, sizeof(word));
        REQUIRE(word == 0x12345678U);
    }
    REQUIRE(std::all_of(result.begin() + 768, result.end(), [](u8 v) { return v == 0xAB; }));
}

TEST_CASE("Metal buffer cache skips copies to null buffers", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer null_buffer(fixture.runtime, VideoCommon::NullBufferParams{});
    REQUIRE(null_buffer.Handle() == nil);

    const std::vector<u8> data = Iota(64);
    fixture.Upload(null_buffer, 0, data);
    fixture.runtime.ClearBuffer(null_buffer, 0, 64, 0);
    fixture.runtime.Finish();

    // A missing vertex buffer is replaced with the zero-filled null buffer.
    fixture.runtime.BindVertexBuffer(3, null_buffer, 128, 64, 16);
    const VertexBufferBinding& binding = fixture.runtime.GetVertexBuffers()[3];
    REQUIRE(binding.buffer != nil);
    REQUIRE(binding.offset == 0);
    REQUIRE(binding.stride == 16);
}

TEST_CASE("Metal buffer cache builds non-indexed quad indices", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);

    fixture.runtime.BindQuadIndexBuffer(Maxwell::PrimitiveTopology::Quads, 10, 8);
    REQUIRE(fixture.runtime.GetIndexBuffer().rewritten_count == 12U);
    REQUIRE(fixture.runtime.GetIndexBuffer().type == MTLIndexTypeUInt32);
    REQUIRE(fixture.ReadIndices<u32>(12) ==
            std::vector<u32>{10, 11, 12, 10, 12, 13, 14, 15, 16, 14, 16, 17});

    fixture.runtime.BindQuadIndexBuffer(Maxwell::PrimitiveTopology::QuadStrip, 0, 6);
    REQUIRE(fixture.runtime.GetIndexBuffer().rewritten_count == 12U);
    REQUIRE(fixture.ReadIndices<u32>(12) ==
            std::vector<u32>{0, 3, 1, 0, 2, 3, 2, 5, 3, 2, 4, 5});

    fixture.runtime.BindQuadIndexBuffer(Maxwell::PrimitiveTopology::Quads, 0, 3);
    REQUIRE(fixture.runtime.GetIndexBuffer().rewritten_count == 0U);
}

TEST_CASE("Metal buffer cache widens 8-bit indices", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);
    const std::vector<u8> indices{9, 9, 9, 0, 1, 2, 0xFF, 254, 3, 9, 9, 9};
    fixture.Upload(buffer, 4, indices);

    // Skipping three indices through `first` puts the first index at an unaligned offset.
    fixture.runtime.BindIndexBuffer(Maxwell::PrimitiveTopology::Triangles,
                                    Maxwell::IndexFormat::UnsignedByte, 3, 6, buffer, 4, 12);
    const IndexBufferBinding& binding = fixture.runtime.GetIndexBuffer();
    REQUIRE(binding.type == MTLIndexTypeUInt16);
    REQUIRE(binding.rewritten_count == 6U);
    REQUIRE(fixture.ReadIndices<u16>(6) == std::vector<u16>{0, 1, 2, 0xFFFF, 254, 3});
}

TEST_CASE("Metal buffer cache assembles indexed quads", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);
    const std::array<u16, 10> indices{100, 20, 21, 22, 23, 30, 31, 32, 33, 100};
    fixture.Upload(buffer, 0,
                   std::span<const u8>(reinterpret_cast<const u8*>(indices.data()),
                                       sizeof(indices)));

    // first = 1 starts the quads 2 bytes into a word.
    fixture.runtime.BindIndexBuffer(Maxwell::PrimitiveTopology::Quads,
                                    Maxwell::IndexFormat::UnsignedShort, 1, 8, buffer, 0,
                                    sizeof(indices));
    const IndexBufferBinding& binding = fixture.runtime.GetIndexBuffer();
    REQUIRE(binding.type == MTLIndexTypeUInt32);
    REQUIRE(binding.rewritten_count == 12U);
    REQUIRE(fixture.ReadIndices<u32>(12) ==
            std::vector<u32>{20, 21, 22, 20, 22, 23, 30, 31, 32, 30, 32, 33});
}

TEST_CASE("Metal buffer cache binds supported index formats directly", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);

    fixture.runtime.BindIndexBuffer(Maxwell::PrimitiveTopology::Triangles,
                                    Maxwell::IndexFormat::UnsignedShort, 4, 6, buffer, 64, 128);
    const IndexBufferBinding& binding = fixture.runtime.GetIndexBuffer();
    REQUIRE(binding.buffer == buffer.Handle());
    REQUIRE(binding.offset == 64);
    REQUIRE(binding.type == MTLIndexTypeUInt16);
    REQUIRE_FALSE(binding.rewritten_count.has_value());
}

TEST_CASE("Metal buffer cache records descriptors in order", "[video_core][metal]") {
    const auto device = MakeDevice();
    if (!device) {
        return;
    }
    Fixture fixture(*device);
    Buffer buffer(fixture.runtime, 0x10000, 0x10000);

    fixture.runtime.BindUniformBuffer(buffer, 0, 256);
    fixture.runtime.BindStorageBuffer(buffer, 256, 512, true);
    const std::span<u8> mapped = fixture.runtime.BindMappedUniformBuffer(0, 0, 64);
    REQUIRE(mapped.size() >= 64);

    const auto descriptors = fixture.runtime.GetDescriptors();
    REQUIRE(descriptors.size() == 3);
    REQUIRE(descriptors[0].kind == BufferDescriptor::Kind::Uniform);
    REQUIRE(descriptors[0].buffer == buffer.Handle());
    REQUIRE(descriptors[1].kind == BufferDescriptor::Kind::Storage);
    REQUIRE(descriptors[1].offset == 256);
    REQUIRE(descriptors[1].is_written);
    REQUIRE(descriptors[2].kind == BufferDescriptor::Kind::Uniform);
    REQUIRE(descriptors[2].size == 64);

    fixture.runtime.ClearDescriptors();
    REQUIRE(fixture.runtime.GetDescriptors().empty());
}
