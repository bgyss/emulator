// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "video_core/host_device_info.h"

class QWindow;

namespace VkDeviceInfo {
/// Fills records with the host GPUs Vulkan can use, probing present modes on a test surface
/// created as a child of window.
void PopulateRecords(std::vector<VideoCore::HostDeviceRecord>& records, QWindow* window);
} // namespace VkDeviceInfo
