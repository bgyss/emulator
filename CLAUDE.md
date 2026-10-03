# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Citron Neo is a Nintendo Switch emulator in C++. It is a fork of the yuzu/Citra lineage, so many files carry yuzu/Citra SPDX headers and conventions. Targets: Linux, Windows, Android and iOS. macOS has partial support (see below).

## Building

Dependencies come from CPM (`-DCITRON_USE_CPM=ON`, cached in `CPM_SOURCE_CACHE`), not from git submodules. The `externals/` submodules are normally left uninitialized and the scripts pass `-DCITRON_CHECK_SUBMODULES=OFF`. Qt is downloaded at configure time through `aqt` (`CMakeModules/qt_download.cmake`, Qt 6.9.3). FFmpeg is built from source with autotools (`externals/ffmpeg/CMakeLists.txt`).

- **Linux:** `./build-citron-linux.sh setup` (once), then `./build-citron-linux.sh use --pgo none --lto full`. The binary lands in `build/use-nopgo/bin/citron`. The PGO flow is `generate` → run the instrumented binary and exit cleanly → `use`. Pass `--nopackage` to skip the AppImage. See `docs/BUILDING-CITRON-LINUX.md`.
- **Windows:** `build-clangtron-windows.sh`, run from an MSYS2 CLANG64 shell, with `--compiler clang-cl` (native) or `llvm-mingw` (also works as a cross-compile from Linux). See `docs/BUILDING-CITRON-WINDOWS.md`.
- **Manual CMake:** to see the full set of flags the scripts use, read the `_CMAKE_ARGS` array in `build-citron-linux.sh` (around line 1140). After the initial configure, rebuild incrementally with `ninja -C <build-dir>`.

### macOS (Apple Silicon)

- `./build-citron-macos.sh setup` installs the Homebrew tools and `aqt`; run it once.
- `./build-citron-macos.sh build` builds `build/macos/bin/citron.app`. Options: `--build-type`, `--build-dir`, `--lto`, `--tests`, `--jobs`.
- `./build-citron-macos.sh package` makes a self-contained bundle in `build/macos/package/`: it runs `macdeployqt`, strips the rpath that points into the CPM cache, ad-hoc signs the app and zips it.
- The script always builds with the `/usr/bin/clang` shims, because a Homebrew or Nix `gcc` on `PATH` can't link against the macOS SDK. It also exports `SDKROOT`, since the autotools sub-builds (OpenSSL, FFmpeg, libusb) don't get CMake's `-isysroot`. The raw toolchain clang that `xcrun --find clang` returns fails in OpenSSL's build with missing `<stdlib.h>`. It also puts Homebrew's `libtool/libexec/gnubin` on `PATH`, since libusb's bootstrap needs the unprefixed `libtoolize`.
- An unpackaged `bin/citron.app` finds Qt through an absolute `LC_RPATH` into the CPM cache, so it only runs on the machine that built it.

How the macOS build differs at runtime:

- MoltenVK 1.4.2 is downloaded and copied into the app bundle (`src/citron/CMakeLists.txt`). `USE_SYSTEM_MOLTENVK=ON` uses the system copy instead.
- The CPU runs on dynarmic's arm64 JIT. NCE (native code execution) is only enabled for Linux and Android (`HAS_NCE` in the root `CMakeLists.txt`).
- Fastmem is off, because `src/common/host_memory.cpp` only has Windows and Linux/FreeBSD implementations and falls back to the generic path.
- Video decode is software-only. Citron has no VideoToolbox path, and NVDEC/VA-API/VDPAU are skipped on Apple.

## Tests and linting

- Tests use Catch2 and build into a single `tests` binary (`src/tests/`). Configure with `-DCITRON_TESTS=ON` (the build scripts turn tests off). Run `<build>/bin/tests`, or pass a Catch2 filter to run a subset, e.g. `<build>/bin/tests "[common]"` or a test name.
- Formatting: run `clang-format` with `src/.clang-format`. CMake creates a `clang-format` target when it finds clang-format (it prefers `clang-format-20`).
- Whitespace policy (`hooks/pre-commit`): no tabs and no trailing whitespace anywhere in `src/` or `CMakeLists.txt`.

## Architecture

The frontends sit on top of a shared core:

- `src/citron/` is the Qt desktop frontend (main window, configuration UI, Qt applets). `src/citron_cmd/` is the SDL2 command-line frontend. `src/android/` and `src/ios/` are the mobile frontends. `src/frontend_common/` holds config shared by all of them.
- `src/core/` is the emulated console:
  - `core.cpp` owns `System`.
  - `hle/kernel/` reimplements the Horizon OS kernel: processes, threads, SVCs and memory.
  - `hle/service/` reimplements system services as IPC servers (am, nvdrv, hid, audio, …).
  - `file_sys/` and `loader/` handle game formats (NSP/XCI/NCA) and the virtual filesystem.
  - `crypto/` handles key derivation (OpenSSL).
  - `arm/` is the CPU interface. Its backends are `arm/dynarmic` (JIT, from `externals/dynarmic`) and `arm/nce` (direct execution on arm64 Linux/Android).
- `src/video_core/` is the emulated Maxwell GPU:
  - `engines/` contains the 3D, compute, DMA and Fermi engines, plus the macro JIT/interpreter in `macro*.cpp`.
  - `host1x/` contains the video decoders, which use FFmpeg.
  - Host-side caches live in `buffer_cache/`, `texture_cache/` and `query_cache/`.
  - The only real renderer is `renderer_vulkan/`; `renderer_null/` is a stub. There is no OpenGL backend. On macOS the renderer runs through MoltenVK, and some code paths check for it specifically.
- `src/shader_recompiler/` translates Maxwell shader bytecode (`frontend/`) into its own IR, runs optimization passes on it (`ir_opt/`), and emits SPIR-V (`backend/`, via `externals/sirit`). Host shaders used internally live in `src/video_core/host_shaders/`, compiled with `glslangValidator`.
- `src/audio_core/`, `src/hid_core/` and `src/input_common/` implement audio (cubeb/SDL), the HID service model, and host input drivers, respectively.
- `src/common/` holds the shared utilities: logging, settings (`settings.h`), host memory, fibers and threading.

CMake setup is split across several modules:

- `CMakeModules/dependencies.cmake` declares the CPM dependencies.
- `CMakeModules/DownloadExternals.cmake` handles the prebuilt downloads (MoltenVK, Qt helpers).
- `BuildClangtronFFmpeg.cmake` and `BuildClangClFFmpeg.cmake` build FFmpeg for the Windows toolchains.
- `openssl_build.cmake` builds OpenSSL.
