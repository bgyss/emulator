# Building Citron Neo for macOS

The macOS build runs inside a [Nix](https://nixos.org) dev shell (`flake.nix`)
and is driven by [mise](https://mise.jdx.dev) tasks (`mise.toml`). Nothing is
installed with Homebrew. It targets Apple Silicon (Intel is untested).

## Prerequisites

- **Xcode** (or the Command Line Tools: `xcode-select --install`). Apple clang
  and the macOS SDK come from here, not from Nix.
- **Nix** with flakes enabled (the Determinate installer enables them).
- **mise**. It is not required to build; you can call `scripts/in-nix` directly.

Nix supplies everything else: cmake, ninja, nasm, glslang, pkg-config,
autoconf, automake, libtool, `aqt` (which downloads Qt), clang-format and git.

## Quick start

```bash
git clone https://github.com/citron-neo/emulator.git
cd emulator

mise trust               # once, to allow this repo's mise.toml
mise run check           # confirm the nix shell provides every tool
mise run build           # -> build/macos/bin/citron.app
mise run run             # launch it
```

The first build downloads Qt, MoltenVK and every CPM dependency, and builds
FFmpeg, OpenSSL and libusb from source, so expect it to take a while.
Downloads are cached in `~/.cache/cpm`.

## Tasks

|Task                |What it does                                               |
|---|---|
|`mise run check`    |Verify the nix shell provides every required tool          |
|`mise run build`    |Build `build/macos/bin/citron.app`                         |
|`mise run build-tests`|Build `citron.app` plus the Catch2 `tests` binary        |
|`mise run test`     |Build with tests, then run `build/macos/bin/tests`         |
|`mise run package`  |Self-contained, ad-hoc signed `.app` and `.zip` in `build/macos/package/`|
|`mise run run`      |Launch the packaged app if present, otherwise the built one|
|`mise run clean`    |Delete the build directory                                 |
|`mise run shell`    |Interactive shell inside the nix dev environment           |

Pass build options after `--`:

```bash
mise run build -- --build-type Debug --jobs 8
mise run build -- --lto
```

|Option                                      |Default       |
|---|---|
|`--build-type <Release\|RelWithDebInfo\|Debug>`|`Release`     |
|`--build-dir <path>`                        |`build/macos` |
|`--lto` / `--no-lto`                        |off           |
|`--tests`                                   |off           |
|`--metal`                                   |off           |
|`--asan` (AddressSanitizer; pair with `--build-dir build/asan`)|off|
|`--jobs <n>`                                |all cores     |

## Without mise

```bash
scripts/in-nix ./build-citron-macos.sh build
```

Avoid a bare `nix develop` in the repo root: Nix would copy the whole tree,
including `build/`, into the store.

`scripts/in-nix` copies `flake.nix` and `flake.lock` into `.dev/nix-shell/`
and enters that flake. This way an untracked flake works and the multi-gigabyte
`build/` tree never gets copied into the Nix store. It does nothing extra when
`CITRON_NIX=1` is already set (that is, you are already inside the shell).
`build-citron-macos.sh` refuses to run outside the shell.

## Why Apple clang and not a Nix compiler

The dev shell is `mkShellNoCC` and the script compiles with `/usr/bin/clang`.
A Nix `gcc` or `clang` on `PATH` cannot link against the macOS SDK, and the
autotools sub-builds (OpenSSL, FFmpeg, libusb) would otherwise pick it up. The
flake's `shellHook` unsets `DEVELOPER_DIR` and `SDKROOT` so `xcrun` finds the
real Xcode SDK, and the script exports `SDKROOT` for the sub-builds that don't
get CMake's `-isysroot`.

## Runtime notes

- CPU emulation uses dynarmic's arm64 JIT. NCE and fastmem are Linux-only.
- Vulkan runs on MoltenVK, which is copied into `Contents/Frameworks`.
- `--metal` (`-DCITRON_ENABLE_METAL=ON`) adds an experimental native Metal
  renderer, selectable as "Metal (Experimental)" in Graphics settings. It only
  presents a cleared frame so far; guest graphics still need the Vulkan path.
- Video decode is software-only (no VideoToolbox path).
- An unpackaged `build/macos/bin/citron.app` finds Qt through an absolute
  rpath into the CPM cache, so it only runs on the machine that built it. Use
  `mise run package` for a portable bundle.
- dynarmic is patched at fetch time by
  `patches/dynarmic-arm64-emit-assert-side-effects.patch`. Without it the arm64
  JIT re-emits every block in Release builds and games appear to hang.

## Troubleshooting

- **`Not inside the nix dev shell`**: run through `mise run ...` or
  `scripts/in-nix ./build-citron-macos.sh ...`.
- **CMake errors about `/opt/homebrew/...` not found**: the build directory was
  configured before the move to Nix and its `CMakeCache.txt` still points at
  Homebrew tools. Either `mise run clean` and rebuild from scratch, or clear
  the stale entries once:
  `scripts/in-nix cmake -UCMAKE_MAKE_PROGRAM -UAUTOCONF -UGLSLANGVALIDATOR -ULIBTOOLIZE -UMAKECOMMAND -UPKG_CONFIG_EXECUTABLE -UVulkan_GLSLANG_VALIDATOR_EXECUTABLE -UCMAKE_EDIT_COMMAND -UFIND_PACKAGE_MESSAGE_DETAILS_PkgConfig -S . -B build/macos`
- **`<stdlib.h>` not found in OpenSSL's build**: something put the raw
  toolchain clang (`xcrun --find clang`) ahead of `/usr/bin/clang`. Don't
  change the script's compiler settings.
