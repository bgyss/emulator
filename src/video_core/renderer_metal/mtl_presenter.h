// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>

#include "common/common_types.h"

namespace Metal {

/// Owns the Metal device and presents to a CAMetalLayer. Kept free of Objective-C types so it can
/// be used from plain C++; the implementation lives in mtl_presenter.mm.
class Presenter {
public:
    /// @param layer CAMetalLayer to present to, or nullptr to run headless.
    /// @throws std::runtime_error when no Metal device is available.
    explicit Presenter(void* layer);
    ~Presenter();

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    /// Clears the next drawable to the background color and presents it.
    void Present(u32 width, u32 height, float red, float green, float blue, bool vsync);

    [[nodiscard]] std::string GetDeviceName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
