# Native Metal Backend Plan

Status as of 2026-10-04. Phase 0 and milestone 2 (present path) are done: the Metal renderer presents guest framebuffers read from guest memory, but the null rasterizer still stands in for the GPU, so GPU-drawn frames stay black. Milestone 3 has started with SPIR-V to MSL translation through SPIRV-Cross (`video_core/renderer_metal/mtl_shader_translator.h`).

A native Metal renderer for Citron Neo mirrors about 26k lines of Vulkan-specific code (`renderer_vulkan/`, `vulkan_common/` and the SPIR-V shader backend); the emulation core and GPU caches are already backend-agnostic.

## Decision first: is native Metal worth it?

Measure where MoltenVK actually loses time before committing; if the gap is mostly shader-compile stutter, a cheaper fix may be enough.

What a native backend buys:

- Metal argument buffers directly, instead of MoltenVK's descriptor emulation.
- Control over when shaders and pipelines compile; MoltenVK converts SPIR-V to MSL at pipeline creation, a major stutter source.
- No MoltenVK workarounds, such as the 16 vertex-buffer cap in `video_core/buffer_cache/buffer_cache_base.h` or disabled robustness in `vulkan_device.cpp`.
- One backend shared by macOS and iOS.

What it costs: a second renderer that must track every change to the Vulkan one.

Alternative to try first: KosmicKrisp, Mesa's Vulkan-on-Metal driver, as a drop-in MoltenVK replacement (approximate, from memory; not checked against current releases).

## Existing seams

Four seams already exist, so the work is adding a backend, not restructuring the core.

| Seam | Where | What it gives Metal |
| --- | --- | --- |
| Renderer selection | `Settings::RendererBackend` (`common/settings_enums.h`), `CreateRenderer` (`video_core/video_core.cpp`) | One switch; `RendererNull` is the minimal reference implementation |
| Templated caches | `TextureCache<P>`, `BufferCache<P>`, `QueryCacheBase<Traits>` | A backend supplies a params struct plus runtime classes |
| Shader IR | `shader_recompiler/` frontend, IR and `ir_opt/` | Everything up to emission is shared; `Profile`/`HostTranslateInfo` carry host quirk flags |
| Metal layer | `citron/qt_common.cpp` (`CAMetalLayer`, `WindowSystemType::Cocoa`), `ios/App/RenderView.swift` | Both frontends already own a `CAMetalLayer` to hand the renderer |

## Phase 0: groundwork (done, verified on Apple Silicon)

Phase 0 makes Metal a selectable backend that boots to a cleared, presented screen, without touching emulation. Seven of eight items landed in bgyss/emulator#2, with build fixes in #3 and #4.

- [x] Append `RendererBackend::Metal` to the settings enum (appending keeps saved configs valid)
- [x] `CITRON_ENABLE_METAL` CMake option, Apple-only, linking Metal and QuartzCore
- [x] `video_core/renderer_metal/` stub: creates an `MTLDevice`, clears and presents the `CAMetalLayer`, null rasterizer underneath
- [x] Wire Metal into `CreateRenderer`, the Qt render widget and the SDL frontend, with a Vulkan fallback where Metal is not built
- [x] Settings UI: list Metal on Apple builds; hide Vulkan-only options (device, present modes) when it is selected
- [x] Remove the `vk::Exception` dependency from `video_core/gpu_thread.cpp`
- [x] Hand the `CAMetalLayer` to the renderer through `WindowSystemInfo`
- [x] Put the frontend's direct Vulkan calls (startup checks, device info, VRAM overlay) behind a backend-neutral interface (`video_core/host_device_info.h`)

The `CAMetalLayer` item also fixed a bug: the SDL frontend passed an `NSView` where a `CAMetalLayer` was expected.

Verified 2026-10-04 on Apple Silicon: with Metal selected, a game boots to a black window with audio and the log shows the `Render.Metal` warning. Booting right after clicking OK in Configure aborted in `RealVfsFilesystem::RefreshReference` during game loading. It was a bug in existing code, not Metal: the system's private filesystem was freed while update files still referenced it. Fixed in bgyss/emulator#4, and an AddressSanitizer build on Apple Silicon showed no memory errors.

## Phase 1: shader translation (critical path)

Start with SPIR-V to MSL through SPIRV-Cross to get pixels on screen, then replace it with a native MSL emitter once rendering is correct.

| Option | Pros | Cons |
| --- | --- | --- |
| A: SPIRV-Cross on the existing SPIR-V output | Reuses `backend/spirv` unchanged; fastest bring-up | Same compile cost and quirks as MoltenVK; fiddly binding remaps |
| B: native `shader_recompiler/backend/msl/` | Direct control of bindings, argument buffers, SIMD-group ops | New emitter, one `emit_msl_*.cpp` per IR category, modelled on yuzu's removed GLSL backend |

Shared hard parts, whichever option:

- Geometry shaders: none in Metal; emulate with a compute pre-pass writing to a buffer, or mesh/object shaders on Metal 3.
- Tessellation: compute kernel for factors plus a post-tessellation vertex function; hull/domain stages need restructuring.
- Transform feedback: emulate with buffer writes from the vertex stage.
- Provoking vertex, logic ops, quads, point size, clip distances: new `Profile` flags plus IR lowering passes in `ir_opt/`.
- Host shaders: the ~50 GLSL files in `video_core/host_shaders/` go through glslang, SPIRV-Cross and `xcrun metal` into a `.metallib` at build time.
- Pipeline cache: persist `MTLBinaryArchive` and generated MSL beside the existing disk shader cache.

## Phase 2: `video_core/renderer_metal/`

The Metal renderer mirrors `renderer_vulkan/` file by file, plugging into the same templated caches.

| Vulkan file(s) | Metal equivalent |
| --- | --- |
| `vulkan_common/vulkan_device.*` | `mtl_device`: GPU family, argument buffer tier, SIMD width, format support table |
| `vk_scheduler`, `vk_master_semaphore`, `vk_command_pool` | Command buffer and encoder management; GPU ticks via `MTLSharedEvent` |
| `vk_staging_buffer_pool`, `vk_buffer_cache*` | Shared-storage `MTLBuffer` staging; unified memory makes `USE_MEMORY_MAPS` cheap |
| `vk_texture_cache*`, `blit_image` | `MTLTexture`, views, blits, format conversion; native ASTC and BC |
| `vk_graphics_pipeline`, `vk_compute_pipeline`, `vk_pipeline_cache`, `fixed_pipeline_state` | `MTLRenderPipelineState`, `MTLDepthStencilState`, encoder dynamic state; new pipeline keys |
| `vk_descriptor_pool`, `vk_update_descriptor` | Tier 2 argument buffers or direct binds; 31 buffer slots per stage |
| `vk_render_pass_cache` | `MTLRenderPassDescriptor` building and an encoder-break policy |
| `vk_query_cache` | Visibility result buffers; some query types stay emulated |
| `vk_fence_manager` | `MTLSharedEvent` fences |
| `vk_rasterizer` | `RasterizerMetal` draw and dispatch entry points |
| `vk_swapchain`, `vk_present_manager`, `vk_blit_screen`, `present/` | `CAMetalLayer` drawables, present thread, filters and AA |
| `maxwell_to_vk` | `maxwell_to_mtl` |
| `vk_compute_pass` | Helper compute passes (uint8 and quad indices, ASTC, MSAA copy, queries) |
| `vk_turbo_mode`, `vk_zbc_clear` | Port, or drop turbo mode |

## Metal feature gaps

Each gap below needs a workaround in the renderer or shader backend; geometry, tessellation and transform feedback carry most of the risk.

| Gap | Workaround |
| --- | --- |
| No geometry shaders | Compute pre-pass or mesh shaders |
| Different tessellation model | Compute factor kernel plus post-tessellation vertex function |
| No transform feedback | Vertex-stage buffer writes |
| No D24S8 on Apple Silicon | Map to D32S8 and fix up copies |
| Primitive restart always on | Rewrite index buffers when a game disables it |
| Implicit hazard tracking instead of layouts/barriers | Untracked resources plus explicit fences only where profiling shows a win |
| Sparse textures only on newer GPUs | Not needed by most emulation paths |
| Different limits (viewports, scissors, vertex attributes) | Gate in `mtl_device`, expose through `Profile`/`HostTranslateInfo` |

## Phases 3–4: frontends, packaging, testing

Packaging changes are small; the real gate is a test harness that compares Metal output against the MoltenVK path.

- Qt: skip copying MoltenVK and probing the Vulkan loader when Metal is the backend (`src/citron/CMakeLists.txt`, `startup_checks.cpp`).
- iOS: make the MoltenVK xcframework in `ios/CMakeLists.txt` optional.
- Build: the `.metallib` step needs `xcrun metal` and `metallib` from Xcode, outside the Nix shell, like clang and the SDK today.
- Validation: Metal API validation (`MTL_DEBUG_LAYER`) on in debug builds.
- Tests: Catch2 tests for `maxwell_to_mtl`; golden tests for the MSL emitter (IR in, compiles with `xcrun metal`).
- Regression: frame-capture comparisons against Vulkan/MoltenVK on a fixed title set.

## Milestones

Milestones 1–4 hold most of the new code; milestone 5 holds most of the risk.

1. Phase 0 plus a stub renderer that clears and presents.
2. Present path and host shaders: the guest framebuffer appears.
3. Buffer and texture caches, basic graphics pipelines via SPIRV-Cross: 2D titles boot.
4. Compute, queries, fences: most 3D titles render.
5. Geometry, tessellation, transform feedback emulation: the correctness long tail.
6. Native MSL backend, binary archives, argument buffers: performance.
