// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <memory>
#include <string>

#include <fmt/format.h>

#include "citron_cmd/emu_window/emu_window_sdl3_vk.h"
#include "common/logging.h"
#include "common/scm_rev.h"

#include <cstring>

#include <SDL3/SDL.h>

EmuWindow_SDL3_VK::EmuWindow_SDL3_VK(InputCommon::InputSubsystem* input_subsystem_,
                                     Core::System& system_, bool fullscreen)
    : EmuWindow_SDL3{input_subsystem_, system_} {
    const std::string window_title = fmt::format("citron {} | {}-{} (Vulkan)", Common::g_build_name,
                                                 Common::g_scm_branch, Common::g_scm_desc);
    render_window = SDL_CreateWindow(window_title.c_str(), Layout::ScreenUndocked::Width,
                                     Layout::ScreenUndocked::Height,
                                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (render_window == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to create the SDL window: {}", SDL_GetError());
        std::exit(EXIT_FAILURE);
    }

    const SDL_PropertiesID window_props = SDL_GetWindowProperties(render_window);
    if (window_props == 0) {
        LOG_CRITICAL(Frontend, "Failed to get information from the window manager: {}",
                     SDL_GetError());
        std::exit(EXIT_FAILURE);
    }

    SetWindowIcon();

    if (fullscreen) {
        Fullscreen();
        ShowCursor(false);
    }

    // The native handles live in the window properties; which ones exist depends on the video
    // driver SDL picked at runtime.
    const char* const video_driver = SDL_GetCurrentVideoDriver();
    const auto is_driver = [video_driver](const char* name) {
        return video_driver != nullptr && std::strcmp(video_driver, name) == 0;
    };

    if (is_driver("windows")) {
        window_info.type = Core::Frontend::WindowSystemType::Windows;
        window_info.render_surface =
            SDL_GetPointerProperty(window_props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    } else if (is_driver("x11")) {
        window_info.type = Core::Frontend::WindowSystemType::X11;
        window_info.display_connection =
            SDL_GetPointerProperty(window_props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        // The X11 window is an integer XID, not a pointer.
        window_info.render_surface = reinterpret_cast<void*>(
            SDL_GetNumberProperty(window_props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    } else if (is_driver("wayland")) {
        window_info.type = Core::Frontend::WindowSystemType::Wayland;
        window_info.display_connection =
            SDL_GetPointerProperty(window_props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        window_info.render_surface =
            SDL_GetPointerProperty(window_props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    } else if (is_driver("cocoa")) {
        window_info.type = Core::Frontend::WindowSystemType::Cocoa;
        // Creating the view converts the window to Metal; the CAMetalLayer backs the Vulkan
        // surface through MoltenVK.
        window_info.render_surface = SDL_Metal_GetLayer(SDL_Metal_CreateView(render_window));
    } else if (is_driver("android")) {
        window_info.type = Core::Frontend::WindowSystemType::Android;
        window_info.render_surface =
            SDL_GetPointerProperty(window_props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
    } else {
        LOG_CRITICAL(Frontend, "Window manager subsystem {} not implemented",
                     video_driver != nullptr ? video_driver : "(unknown)");
        std::exit(EXIT_FAILURE);
    }

    OnResize();
    OnMinimalClientAreaChangeRequest(GetActiveConfig().min_client_area_size);
    SDL_PumpEvents();
    LOG_INFO(Frontend, "citron Version: {} | {}-{} (Vulkan)", Common::g_build_name,
             Common::g_scm_branch, Common::g_scm_desc);
}

EmuWindow_SDL3_VK::~EmuWindow_SDL3_VK() = default;

std::unique_ptr<Core::Frontend::GraphicsContext> EmuWindow_SDL3_VK::CreateSharedContext() const {
    return std::make_unique<DummyContext>();
}
