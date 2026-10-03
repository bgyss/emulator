{
  description = "Citron Neo macOS build environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs = { nixpkgs, ... }:
    let
      # Only macOS is covered here. Linux and Windows builds have their own
      # scripts (build-citron-linux.sh, build-clangtron-windows.sh).
      systems = [ "aarch64-darwin" "x86_64-darwin" ];
    in {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let pkgs = import nixpkgs { inherit system; };
        in { default = pkgs.mkShellNoCC {
          # mkShellNoCC on purpose: the build must use Apple clang from
          # /usr/bin and the system SDK. Nix's cc-wrapper on PATH cannot link
          # against the macOS SDK, and the autotools sub-builds (OpenSSL,
          # FFmpeg, libusb) pick it up ahead of the Apple toolchain.
          packages = with pkgs; [
            cmake ninja nasm glslang pkg-config
            autoconf automake libtool   # libusb bootstrap needs unprefixed libtoolize
            aqtinstall                  # downloads Qt at configure time
            clang-tools                 # clang-format
            git perl python3 mise
          ];
          shellHook = ''
            export CITRON_NIX=1
            # Nix's Apple SDK setup hook points these at a store SDK. The build
            # wants the real Xcode toolchain, so let xcrun resolve it.
            unset DEVELOPER_DIR SDKROOT MACOSX_DEPLOYMENT_TARGET
            unset NIX_CFLAGS_COMPILE NIX_LDFLAGS NIX_CFLAGS_LINK NIX_HARDENING_ENABLE
          '';
        }; });
    };
}
