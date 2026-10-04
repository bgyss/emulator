// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <QWindow>

#include "citron/qt_common.h"
#include "citron/vk_device_info.h"
#include "common/settings_enums.h"

namespace VkDeviceInfo {
void PopulateRecords(std::vector<VideoCore::HostDeviceRecord>& records, QWindow* window) {
    // Create a test window with a Vulkan surface type for checking present modes.
    QWindow test_window(window);
    test_window.setSurfaceType(QWindow::VulkanSurface);
    test_window.create();
    const auto wsi = QtCommon::GetWindowSystemInfo(&test_window);

    records = VideoCore::EnumerateHostDevices(Settings::RendererBackend::Vulkan, wsi);
}
} // namespace VkDeviceInfo
