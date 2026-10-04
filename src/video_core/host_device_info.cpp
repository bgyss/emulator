// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging.h"
#include "video_core/host_device_info.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {

namespace {

std::optional<Settings::VSyncMode> PresentModeToVSyncMode(VkPresentModeKHR mode) {
    switch (mode) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return Settings::VSyncMode::Immediate;
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return Settings::VSyncMode::Mailbox;
    case VK_PRESENT_MODE_FIFO_KHR:
        return Settings::VSyncMode::Fifo;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return Settings::VSyncMode::FifoRelaxed;
    default:
        return std::nullopt;
    }
}

std::vector<HostDeviceRecord> EnumerateVulkanDevices(
    const Core::Frontend::EmuWindow::WindowSystemInfo& window_info) try {
    using namespace Vulkan;

    vk::InstanceDispatch dld;
    const auto library = OpenLibrary();
    const vk::Instance instance =
        CreateInstance(*library, dld, VK_API_VERSION_1_1, window_info.type);
    const std::vector<VkPhysicalDevice> physical_devices = instance.EnumeratePhysicalDevices();
    vk::SurfaceKHR surface = CreateSurface(instance, window_info);

    std::vector<HostDeviceRecord> records;
    records.reserve(physical_devices.size());
    for (const VkPhysicalDevice device : physical_devices) {
        const auto physical_device = vk::PhysicalDevice(device, dld);

        HostDeviceRecord record;
        record.name = physical_device.GetProperties().deviceName;
        for (const VkPresentModeKHR mode : physical_device.GetSurfacePresentModesKHR(*surface)) {
            if (const auto vsync_mode = PresentModeToVSyncMode(mode)) {
                record.vsync_modes.push_back(*vsync_mode);
            }
        }

        VkPhysicalDeviceDriverProperties driver_properties{};
        driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        driver_properties.pNext = nullptr;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        properties.pNext = &driver_properties;
        dld.vkGetPhysicalDeviceProperties2(physical_device, &properties);
        record.has_broken_compute = Device::CheckBrokenCompute(
            driver_properties.driverID, properties.properties.driverVersion);

        records.push_back(std::move(record));
    }
    return records;
} catch (const Vulkan::vk::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Failed to enumerate devices with error: {}", exception.what());
    return {};
}

std::optional<std::string> CheckVulkan() try {
    Vulkan::vk::InstanceDispatch dld;
    const auto library = Vulkan::OpenLibrary();
    const Vulkan::vk::Instance instance =
        Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1);
    return std::nullopt;
} catch (const Vulkan::vk::Exception& exception) {
    return std::string{exception.what()};
}

} // Anonymous namespace

std::vector<HostDeviceRecord> EnumerateHostDevices(
    Settings::RendererBackend backend,
    const Core::Frontend::EmuWindow::WindowSystemInfo& window_info) {
    switch (backend) {
    case Settings::RendererBackend::Vulkan:
        return EnumerateVulkanDevices(window_info);
    default:
        return {};
    }
}

std::optional<std::string> CheckHostApi(Settings::RendererBackend backend) {
    switch (backend) {
    case Settings::RendererBackend::Vulkan:
        return CheckVulkan();
    default:
        return std::nullopt;
    }
}

} // namespace VideoCore
