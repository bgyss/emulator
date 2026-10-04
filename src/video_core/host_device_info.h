// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "common/settings_enums.h"
#include "core/frontend/emu_window.h"

namespace VideoCore {

/// A host GPU as the frontend's settings UI needs to see it, independent of the graphics API.
struct HostDeviceRecord {
    std::string name;
    /// VSync modes the device can present with on the probed surface.
    std::vector<Settings::VSyncMode> vsync_modes;
    /// The driver is known to break compute shaders, so the compute option should be offered.
    bool has_broken_compute{};
};

/**
 * Lists the host GPUs the given backend can render with, in the order the backend's device
 * setting indexes them. Returns an empty list for backends without device selection or when
 * enumeration fails (the failure is logged).
 *
 * @param window_info A surface to query presentation support against.
 */
[[nodiscard]] std::vector<HostDeviceRecord> EnumerateHostDevices(
    Settings::RendererBackend backend,
    const Core::Frontend::EmuWindow::WindowSystemInfo& window_info);

/**
 * Starts the backend's driver loader to check that the host API is usable.
 *
 * @returns An error message when it is not, std::nullopt otherwise.
 */
[[nodiscard]] std::optional<std::string> CheckHostApi(Settings::RendererBackend backend);

} // namespace VideoCore
