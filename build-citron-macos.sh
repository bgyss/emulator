#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# build-citron-macos.sh — Local macOS build script
#
# Lives in the citron repo root. Like build-citron-linux.sh, it fetches every
# library dependency through CPM at configure time (no submodule init needed),
# downloads Qt through aqt and builds FFmpeg from source.
#
# Usage:
#   ./build-citron-macos.sh setup              # once per machine
#   ./build-citron-macos.sh build [options]    # -> build/macos/bin/citron.app
#   ./build-citron-macos.sh package [options]  # self-contained .app + .zip
#   ./build-citron-macos.sh run [options]      # launch the built app
#   ./build-citron-macos.sh clean [options]    # delete the build directory
#
# Options:
#   --build-type <Release|RelWithDebInfo|Debug>   (default: Release)
#   --build-dir <path>                            (default: build/macos)
#   --lto / --no-lto      Link-time optimization  (default: off)
#   --tests               Also build the Catch2 `tests` binary
#   --jobs <n>            Parallel build jobs     (default: all cores)
#
# Runtime notes (Apple Silicon):
#   - CPU emulation uses dynarmic's arm64 JIT; NCE and fastmem are Linux-only.
#   - Vulkan runs on the bundled MoltenVK (copied into Contents/Frameworks).
#   - Video decode is software-only (no VideoToolbox path).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*" >&2; }
success() { echo -e "${GREEN}[OK]${RESET} $*" >&2; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*" >&2; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}${GREEN}== $* ==${RESET}" >&2; }

usage() { sed -n '/^# Usage:/,/^# Runtime notes/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

[[ "$(uname -s)" == "Darwin" ]] || error "This script only runs on macOS. Use build-citron-linux.sh on Linux."

# ── Defaults ─────────────────────────────────────────────────────────────────
STAGE="${1:-}"
[[ -n "${STAGE}" ]] && shift
BUILD_TYPE="Release"
BUILD_DIR="build/macos"
LTO="OFF"
TESTS="OFF"
JOBS="$(sysctl -n hw.logicalcpu)"
CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-${HOME}/.cache/cpm}"
HOST_ARCH="$(uname -m)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-type) BUILD_TYPE="${2:?--build-type needs a value}"; shift 2 ;;
        --build-dir)  BUILD_DIR="${2:?--build-dir needs a value}"; shift 2 ;;
        --lto)        LTO="ON"; shift ;;
        --no-lto)     LTO="OFF"; shift ;;
        --tests)      TESTS="ON"; shift ;;
        --jobs)       JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) error "Unknown argument: $1\nRun with --help for usage." ;;
    esac
done

case "${BUILD_TYPE}" in
    Release|RelWithDebInfo|Debug) ;;
    *) error "--build-type must be Release, RelWithDebInfo or Debug (got '${BUILD_TYPE}')" ;;
esac

APP_PATH="${BUILD_DIR}/bin/citron.app"
PACKAGE_DIR="${BUILD_DIR}/package"

# Homebrew installs GNU libtool as glibtool/glibtoolize; libusb's bootstrap
# needs the unprefixed names. aqt (from uv/pipx) lands in ~/.local/bin.
setup_env() {
    local brew_prefix
    brew_prefix="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    export PATH="${brew_prefix}/opt/libtool/libexec/gnubin:${HOME}/.local/bin:${brew_prefix}/bin:${PATH}"
    # Always use Apple clang: a Homebrew/Nix gcc earlier on PATH can't link
    # against the macOS SDK. Use the /usr/bin xcrun shims (not the raw toolchain
    # binaries) and export SDKROOT so autotools sub-builds that bypass CMake's
    # -isysroot (OpenSSL, FFmpeg, libusb) still find the SDK headers.
    CC_BIN="/usr/bin/clang"
    CXX_BIN="/usr/bin/clang++"
    SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    export SDKROOT
}

# ── setup ────────────────────────────────────────────────────────────────────
BREW_PACKAGES=(cmake ninja nasm glslang pkg-config autoconf automake libtool)

stage_setup() {
    header "Setting up macOS build dependencies"

    if ! xcode-select -p >/dev/null 2>&1; then
        info "Installing Xcode Command Line Tools (follow the dialog, then re-run setup)"
        xcode-select --install || true
        exit 1
    fi
    success "Xcode toolchain: $(xcode-select -p)"

    command -v brew >/dev/null || error "Homebrew is required: https://brew.sh"
    info "Installing Homebrew packages: ${BREW_PACKAGES[*]}"
    brew install "${BREW_PACKAGES[@]}"

    setup_env
    if command -v aqt >/dev/null; then
        success "aqt already installed: $(command -v aqt)"
    elif command -v uv >/dev/null; then
        uv tool install aqtinstall
    elif command -v pipx >/dev/null; then
        pipx install aqtinstall
    else
        info "Installing pipx to provide aqt (Qt downloader)"
        brew install pipx
        pipx install aqtinstall
    fi

    stage_check
}

stage_check() {
    setup_env
    local ok=1 tool
    for tool in cmake ninja nasm glslangValidator pkg-config autoconf automake libtoolize aqt git perl python3; do
        if command -v "${tool}" >/dev/null; then
            success "  ${tool} -> $(command -v "${tool}")"
        else
            warn "  ${tool} -> NOT FOUND"; ok=0
        fi
    done
    [[ ${ok} -eq 1 ]] || error "Missing tools. Run: ./build-citron-macos.sh setup"
}

# ── build ────────────────────────────────────────────────────────────────────
stage_build() {
    header "Building citron (${BUILD_TYPE}, ${HOST_ARCH}, LTO=${LTO}) in ${BUILD_DIR}"
    stage_check

    local cmake_args=(
        -S . -B "${BUILD_DIR}" -G Ninja
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DCMAKE_C_COMPILER=${CC_BIN}"
        "-DCMAKE_CXX_COMPILER=${CXX_BIN}"
        "-DCMAKE_OSX_ARCHITECTURES=${HOST_ARCH}"
        "-DCITRON_USE_CPM=ON"
        "-DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}"
        "-DCITRON_USE_BUNDLED_VCPKG=OFF"
        "-DCITRON_USE_BUNDLED_QT=ON"
        "-DUSE_SYSTEM_QT=OFF"
        "-DCITRON_USE_BUNDLED_FFMPEG=ON"
        "-DBUILD_TESTING=${TESTS}"
        "-DCITRON_TESTS=${TESTS}"
        "-DCITRON_DOWNLOAD_TIME_ZONE_DATA=ON"
        "-DCITRON_CHECK_SUBMODULES=OFF"
        "-DCITRON_USE_LLVM_DEMANGLE=OFF"
        "-DCITRON_USE_QT_WEB_ENGINE=OFF"
        "-DCITRON_USE_QT_MULTIMEDIA=OFF"
        "-DQT_NO_PRIVATE_MODULE_WARNING=ON"
        "-DENABLE_QT_TRANSLATION=ON"
        "-DENABLE_WEB_SERVICE=ON"
        "-DENABLE_OPENSSL=ON"
        "-DCITRON_USE_FASTER_LD=OFF"
        "-DCITRON_ENABLE_LTO=${LTO}"
        "-DCITRON_BUILD_TYPE=Release"
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    )

    info "Configuring (the first run downloads Qt, MoltenVK and all CPM dependencies)"
    cmake "${cmake_args[@]}"

    info "Compiling with ${JOBS} jobs"
    cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

    [[ -d "${APP_PATH}" ]] || error "Build finished but ${APP_PATH} is missing"
    success "Built ${APP_PATH}"
    [[ "${TESTS}" == "ON" ]] && success "Tests: ${BUILD_DIR}/bin/tests"
    info "Run it with: ./build-citron-macos.sh run --build-dir ${BUILD_DIR}"
}

# ── package ──────────────────────────────────────────────────────────────────
# Produces a self-contained bundle: Qt frameworks/plugins copied in by
# macdeployqt, the build-machine rpath into the CPM cache removed, and an
# ad-hoc signature so Apple Silicon will run it.
stage_package() {
    [[ -d "${APP_PATH}" ]] || stage_build
    setup_env
    header "Packaging ${APP_PATH}"

    local qt_root macdeployqt exe out_app zip_name
    qt_root="$(sed -n 's/^QT_TARGET_PATH:PATH=//p' "${BUILD_DIR}/CMakeCache.txt")"
    macdeployqt="${qt_root}/bin/macdeployqt"
    [[ -x "${macdeployqt}" ]] || error "macdeployqt not found at ${macdeployqt}"

    rm -rf "${PACKAGE_DIR}"
    mkdir -p "${PACKAGE_DIR}"
    ditto "${APP_PATH}" "${PACKAGE_DIR}/citron.app"
    out_app="${PACKAGE_DIR}/citron.app"
    exe="${out_app}/Contents/MacOS/citron"

    info "Running macdeployqt"
    "${macdeployqt}" "${out_app}" -verbose=1

    local rpath
    while read -r rpath; do
        if [[ "${rpath}" == /* ]]; then
            info "Removing build-machine rpath ${rpath}"
            install_name_tool -delete_rpath "${rpath}" "${exe}"
        fi
    done < <(otool -l "${exe}" | awk '/LC_RPATH/{getline; getline; print $2}')

    info "Ad-hoc code signing"
    codesign --force --deep --sign - "${out_app}"
    codesign --verify --deep --strict "${out_app}"

    zip_name="citron-macos-${HOST_ARCH}-$(git rev-parse --short HEAD).zip"
    ditto -c -k --keepParent "${out_app}" "${PACKAGE_DIR}/${zip_name}"

    success "App: ${out_app}"
    success "Zip: ${PACKAGE_DIR}/${zip_name}"
}

# ── run / clean ──────────────────────────────────────────────────────────────
stage_run() {
    local app="${APP_PATH}"
    [[ -d "${PACKAGE_DIR}/citron.app" ]] && app="${PACKAGE_DIR}/citron.app"
    [[ -d "${app}" ]] || error "No app at ${app}. Run: ./build-citron-macos.sh build"
    info "Launching ${app}"
    exec "${app}/Contents/MacOS/citron"
}

stage_clean() {
    [[ -n "${BUILD_DIR}" && "${BUILD_DIR}" != "/" && "${BUILD_DIR}" != "." ]] || error "Refusing to clean '${BUILD_DIR}'"
    info "Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    success "Cleaned"
}

case "${STAGE}" in
    setup)   stage_setup   ;;
    check)   stage_check   ;;
    build)   stage_build   ;;
    package) stage_package ;;
    run)     stage_run     ;;
    clean)   stage_clean   ;;
    ""|-h|--help|help) usage ;;
    *) error "Unknown stage: ${STAGE}\nRun with --help for usage." ;;
esac
