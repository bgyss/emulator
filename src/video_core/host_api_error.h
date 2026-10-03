// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <exception>

namespace VideoCommon {

/// Base class for errors raised by a host graphics API (Vulkan, Metal), so code outside the
/// renderers can handle them without depending on a specific API.
class HostApiError : public std::exception {
public:
    /// True when the host driver ran out of device or host memory.
    [[nodiscard]] virtual bool IsOutOfMemory() const noexcept = 0;
};

} // namespace VideoCommon
