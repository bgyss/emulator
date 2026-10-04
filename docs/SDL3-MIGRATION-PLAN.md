# Plan: migrate Citron Neo from SDL2 to SDL3

SDL3 is not source-compatible with SDL2. Almost every API was renamed, and the
audio and window-system-info APIs were redesigned. SDL is used in only four
areas (about 9 source files, roughly 2,000 lines), so the migration is
tractable. This plan is based on a survey of the code; no migration work has
been done yet.

Android builds with `-DENABLE_SDL2=0` and iOS is unaffected, so neither is in
scope.

## Where SDL is used today

| Area | Files | Size of change |
|---|---|---|
| Input (gamepad/joystick/HID) | `src/input_common/drivers/sdl_driver.{h,cpp}` (about 220 SDL refs), `joycon*.{h,cpp}`, `helpers/joycon_driver.*`, `helpers/joycon_protocol/*` (`SDL_hid_*`) | Largest |
| Audio sink | `src/audio_core/sink/sdl2_sink.cpp` (queue-based `SDL_OpenAudioDevice`) | Rewrite |
| CLI frontend window | `src/citron_cmd/emu_window/emu_window_sdl2{,_vk,_null}.*`, `citron_sdl_config.cpp` | Medium |
| Qt frontend | `src/citron/main.cpp` (screensaver inhibit and hints only) | Small |
| Build | root `CMakeLists.txt`, `CMakeModules/dependencies.cmake`, `externals/CMakeLists.txt`, `CopyCitronSDLDeps.cmake`, `src/*/CMakeLists.txt`, the MinGW cross files, `build-*.sh` | Medium |

## Phase 0: decisions (settled)

1. **Hard cut to SDL3.** No SDL2 compatibility shim; the SDL2 code paths are removed.
2. **Version: `release-3.4.18`**, built from source through CPM as a static library on every
   desktop toolchain (including MSVC, which no longer uses prebuilt bundled SDL2 binaries).
3. **Options renamed**: `ENABLE_SDL2` -> `ENABLE_SDL3`, `CITRON_USE_EXTERNAL_SDL2` ->
   `CITRON_USE_EXTERNAL_SDL3`, `HAVE_SDL2` -> `HAVE_SDL3`. `CITRON_USE_BUNDLED_SDL2` and
   `CopyCitronSDLDeps.cmake` are removed.

Backward compatibility kept on purpose:

- The saved audio engine value `"sdl2"` still loads (mapped to the SDL3 sink); new configs write
  `"sdl3"`.
- The CLI config file keeps its `sdl2-config.ini` name so existing settings are not lost.

## Implementation status

Done (syntax-checked against the SDL3 3.4.18 headers; not yet built or run end to end):

- Build system: CMake, CPM pin, externals, MinGW cross files, Windows script package name.
- Audio sink rewritten on `SDL_OpenAudioDeviceStream`.
- `citron-cmd` window: new event model, fullscreen modes, window properties for the Vulkan
  surface, IO streams for the icon.
- Input driver: `SDL_Gamepad` API, ID-based enumeration, binding lookup via
  `SDL_GetGamepadBindings`, new battery and sensor events, updated hints.
- Joy-Con HID code and Qt `main.cpp`: include paths only (the `SDL_hid_*` and screensaver APIs
  are unchanged).

Not done / needs a human with hardware or a full toolchain:

- Full builds on Linux, Windows (clang-cl and llvm-mingw) and macOS, and the Catch2 suite.
- Controller testing (see Phase 3), audio latency/underrun testing, window behaviour on
  X11/Wayland/Windows/macOS.
- `build-citron-linux.sh` still carries SDL2-specific comments about configure-time checks
  (`Xext.h`, ALSA/Pulse); they need re-verifying against SDL3's CMake.
- The Windows script's duplicate-symbol linker workarounds (`__cpuidex`,
  `--allow-multiple-definition`, the `SDL_*_REAL` strip regex) may no longer be needed.

## Phase 1: build system

- Use `find_package(SDL3)` and the `SDL3::SDL3` target. Rework the alias block
  at `CMakeLists.txt:61-65` (`SDL3::SDL3-static` / `SDL3::SDL3-shared`).
- Update the CPM entry in `CMakeModules/dependencies.cmake` (new `GIT_TAG` and
  SDL3 option names). Revisit the SDL2-era subsystem-off list in
  `externals/CMakeLists.txt`.
- Verify `SDL_HIDAPI_LIBUSB` still exists and what its default is. Joy-Con
  support depends on it.
- Windows MSVC bundled binaries (`SDL2-2.28.2`, `CITRON_USE_BUNDLED_SDL2`):
  find an SDL3 prebuilt package, or drop the bundled path and use CPM like the
  clang toolchains. Update `CopyCitronSDLDeps.cmake` to copy `SDL3.dll`.
- Update the MinGW cross files, `build-clangtron-windows.sh` (the
  `mingw-w64-clang-x86_64-SDL2` package becomes SDL3) and the Linux dependency
  notes in `build-citron-linux.sh`.
  - The Windows script has link workarounds for SDL2 duplicate symbols
    (`__cpuidex`, `SDL_*_REAL`); check whether they are still needed.
  - The Linux script has X11/ALSA/Pulse notes tied to SDL2 configure checks.
    SDL3 also has Wayland, PipeWire and libdecor dependencies.
- Update `docs/BUILDING-CITRON-LINUX.md`, and check `flake.nix` and `mise.toml`
  (macOS) for SDL package names.

## Phase 2: code migration (one subsystem per commit)

### Mechanical renames (all areas)

- Headers become `<SDL3/SDL.h>`.
- Booleans: `SDL_TRUE/FALSE` becomes `true/false`, and many functions now
  return `bool` instead of `0` on success. Every `if (SDL_Init(...) < 0)`-style
  check must be inverted.
- Events are renamed (`SDL_KEYDOWN` becomes `SDL_EVENT_KEY_DOWN`, window events
  are flattened, and so on).
- Hints get new names and some are removed (for example
  `SDL_HINT_ACCELEROMETER_AS_JOYSTICK`).

### Input (`sdl_driver.cpp`, the biggest piece)

- `SDL_GameController*` becomes `SDL_Gamepad*`; the `SDL_CONTROLLER_BUTTON_*`,
  `AXIS_*` and `TYPE_*` enums become `SDL_GAMEPAD_*`.
- Enumeration changes: `SDL_NumJoysticks` plus index-based open becomes
  `SDL_GetJoysticks()` returning an ID array. The `SDL_JOYDEVICEADDED` index
  semantics differ, so the add/remove handling needs care.
- GUIDs: `SDL_JoystickGUID` becomes `SDL_GUID`. Strings should stay stable, but
  verify against saved user configs, which could otherwise lose controller
  mappings.
- `SDL_GameControllerGetBindForButton` becomes `SDL_GetGamepadBindings`, which
  changes how bind-type and hat logic is consumed.
- Sensors move to `SDL_SENSOR_*` with `SDL_SetGamepadSensorEnabled`.
- Rumble: `SDL_JoystickRumble` remains; check the `SDL_Vibration` types.
- Battery: `SDL_JoystickPowerLevel` becomes `SDL_GetJoystickPowerInfo` (returns
  a percentage) and the `SDL_JOYSTICK_POWER_*` enums become `SDL_POWERSTATE_*`.
- The SDL3 Joy-Con HIDAPI driver has new hints and is stricter about
  combining. Re-test `COMBINE_JOY_CONS` and `VERTICAL_JOY_CONS`.
- `SDL_hid_*` is mostly a rename (now `SDL3/SDL_hidapi.h`, with a slightly
  different struct). Touches `joycon*` and `joycon_protocol/*`.

### Audio sink (`sdl2_sink.cpp`)

A real redesign, not a rename.

- SDL3 removes `SDL_OpenAudioDevice`, `SDL_QueueAudio` and
  `SDL_ClearQueuedAudio`. The replacement is `SDL_OpenAudioDeviceStream` with
  `SDL_PutAudioStreamData`, or a get-callback.
- Enumeration changes to `SDL_GetAudioPlaybackDevices` and
  `SDL_GetAudioDeviceName(id)`.
- Pause/resume become `SDL_PauseAudioDevice` / `SDL_ResumeAudioDevice`.
- Review the sink's current pull/queue model and measure latency before and
  after.

### CLI window (`emu_window_sdl2*.cpp`)

- `SDL_CreateWindow` is now `(title, w, h, flags)`; the
  `SDL_WINDOWPOS_UNDEFINED` args are gone. Use
  `SDL_CreateWindowWithProperties` for advanced setup.
- `SDL_GetWindowWMInfo` and `SDL_SysWMinfo` are removed. Native handles come
  from `SDL_GetWindowProperties` with `SDL_PROP_WINDOW_X11_*`,
  `WAYLAND_*`, `WIN32_HWND_POINTER`, `COCOA_*`. `emu_window_sdl2_vk.cpp` needs
  a rewrite of its window-type switch.
- RW/surface: `SDL_RWFromConstMem` becomes `SDL_IOFromConstMem`,
  `SDL_LoadBMP_RW` becomes `SDL_LoadBMP_IO`, `SDL_FreeSurface` becomes
  `SDL_DestroySurface`.
- `SDL_GL_GetDrawableSize` and `SDL_WINDOW_ALLOW_HIGHDPI` become
  `SDL_GetWindowSizeInPixels` and `SDL_WINDOW_HIGH_PIXEL_DENSITY`.
- Fullscreen: `SDL_WINDOW_FULLSCREEN_DESKTOP` is removed; use
  `SDL_SetWindowFullscreen(win, bool)`.
- Cursor: `SDL_ShowCursor` splits into `SDL_ShowCursor` / `SDL_HideCursor`.
- Touch events become `SDL_EVENT_FINGER_*` with normalised coordinates, and
  mouse events changed.
- Scancodes in `citron_sdl_config.cpp` mostly keep their names.
  `SDL_SetMainReady` and `SDL_MAIN_HANDLED` remain, but main handling moved to
  `<SDL3/SDL_main.h>`; check how the cmd binary's `main` is set up.
- `SDL_GetTicks` returns `Uint64`; `SDL_GetDesktopDisplayMode` takes a display
  ID.

### Qt frontend (`citron/main.cpp`)

Screensaver calls keep their names, but `SDL_InitSubSystem` now returns `bool`
and the screensaver hints (`SDL_HINT_VIDEO_ALLOW_SCREENSAVER`,
`SDL_HINT_SCREENSAVER_INHIBIT_ACTIVITY_NAME`) are renamed or replaced.

## Phase 3: verification

- Build matrix: Linux (including the PGO flow), Windows clang-cl and
  llvm-mingw, and macOS via `mise run build`. Run `mise run test` and
  `<build>/bin/tests` (Catch2).
- Manual testing:
  - Controllers: Xbox, PS4/5, Switch Pro, Joy-Con (single and combined), plus
    rumble, gyro and battery readout.
  - Audio: playback, device selection and underruns.
  - `citron-cmd`: window creation under X11, Wayland, Windows and macOS
    (MoltenVK Metal layer), plus fullscreen, resize and touch.
  - Screensaver inhibit in the Qt app.
- Config compatibility: load an existing user config with saved SDL input GUIDs
  and confirm the profiles still map.
- Run `clang-format` and the whitespace pre-commit hook.

## Risks (roughly in order)

1. The audio sink rewrite (behaviour and latency change, no mechanical path).
2. Controller GUID and index semantics, which could break saved bindings and
   hot-plug handling.
3. Windows and macOS packaging; the bundled-binary path and linker workarounds
   are the most fragile build code.
4. The Wayland/X11 native-handle lookup in the Vulkan window path.
5. Distro packaging: SDL3 is not in older distro repos, so the CPM/static route
   stays necessary.

## Suggested commit order

1. Build system, plus a temporary shim so everything still compiles.
2. The mechanical rename pass.
3. The cmd window and Vulkan surface.
4. Audio.
5. Input.
6. Joy-Con HID.
7. Qt `main.cpp`.
8. Scripts and docs cleanup.
