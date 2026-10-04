# Metal Renderer Handoff

Status as of 2026-10-04, after PR #18. This is a handoff for whoever picks the native Metal renderer up next. It covers what works, how to build and check it, the known gaps in priority order, and the traps already hit. The original design and phase plan are in `docs/METAL-BACKEND-PLAN.md`.

## Where it stands

Metroid Dread boots and plays on an M1 Max with the native Metal renderer, with correct colors and orientation. All 37 `[metal]` Catch2 tests pass on that machine.

Implemented, all in `src/video_core/renderer_metal/`:

| Area | Files | Notes |
| --- | --- | --- |
| Device, scheduler, staging | `mtl_device`, `mtl_scheduler`, `mtl_staging_buffer_pool` | One command buffer at a time. Encoders are opened lazily and reused while compatible. |
| Buffer cache runtime | `mtl_buffer_cache*` | Private-storage buffers. Blit copies and clears. Compute kernels widen 8-bit indices and turn quads into triangles. |
| Texture cache runtime | `mtl_texture_cache*`, `maxwell_to_mtl` | Covers the full guest format table, views, uploads/downloads, copies, scaled blits, samplers and framebuffers. 24-bit depth is stored as `Depth32Float(_Stencil8)`, converted by compute kernels on copy. |
| Clears | `mtl_clear_helper` | A load-action clear when the clear covers the whole attachment, otherwise a scissored draw. |
| Shaders | `mtl_shader_translator`, `mtl_pipeline_cache` | Shader recompiler → SPIR-V → SPIRV-Cross → MSL → `newLibraryWithSource`. |
| Graphics pipelines and draws | `mtl_graphics_pipeline`, `mtl_rasterizer` | The pipeline key reuses Vulkan's `FixedPipelineState`. Per-draw state is set on the encoder. |
| Shader stutter | `mtl_pipeline_cache`, `mtl_graphics_pipeline` | Pipelines can compile on background workers (`use_asynchronous_shaders`), and there's a per-title disk cache (`metal.bin`, `metal_states.bin`). |
| Present | `mtl_presenter`, `renderer_metal` | Presents the GPU-rendered texture straight from the texture cache, falling back to a CPU read. |

### Measured stutter numbers (Dread, M1 Max)

- **Second launch:** 474 pipelines loaded from the disk cache in 0.3 s. macOS's own Metal shader cache already had the compiled code.
- **New content during play:** about 200 ms per brand-new MSL function in Apple's compiler, plus about 16 ms per pipeline state. With async shaders on, that shows up as brief pop-in of new effects: 543 skipped draws across 39 new pipelines. Our own steps are cheap: recompile 0.7 ms per pipeline, MSL translation 2.4 ms per function.

## Building and checking it

### Building with Metal

Metal is **opt-in**: `build-citron-macos.sh` passes `-DCITRON_ENABLE_METAL=OFF` unless it gets `--metal`. It passes the flag on every configure, so a build without `--metal`, including `mise run test` and plain `mise run build`, silently switches Metal off. The Metal tests then disappear from `tests` ("No test cases matched '[metal]'").

```sh
mise run build -- --metal --tests
build/macos/bin/tests "[metal]"   # quote it: zsh treats [metal] as a glob
mise run run                      # then pick Metal as the graphics API
```

A quick fix worth doing early: make the mise tasks pass `--metal`, or default `METAL=ON` on macOS.

### Logs

- **Log file:** `~/.local/share/citron/log/citron_log.txt`. It is replaced on every launch.
- **Log filter:** the default is `*:Warning`, which hides the pipeline statistics. Set it to `*:Warning Render.Metal:Info` to see them.
- **Pipeline statistics:** every 100 pipelines, after loading the disk cache, and at shutdown.
- **Missing features:** each logs once at Warning level, so this summarizes what a game hit:

  ```sh
  grep "Render.Metal" ~/.local/share/citron/log/citron_log.txt | sed -E 's/^\[ *[0-9.]+\] //' | sort | uniq -c | sort -rn | head -40
  ```

### Checking changes without a Mac

The PRs so far were written on Linux and checked in two ways:

- **Syntax check:** `clang++ -x objective-c++ -fobjc-arc -fblocks -fobjc-runtime=gnustep-2.0 -std=c++20 -fsyntax-only` against hand-written stub `Metal/Metal.h` and `TargetConditionals.h` headers.
- **Translator harness:** a Linux build of the translator against real SPIRV-Cross.

The stubs were never committed, so they would need to be recreated. They don't catch MSL errors inside runtime shader strings: compile those on a Mac or emulate the logic in C++. Real verification only happens with `tests "[metal]"` and a game on a Mac.

## Remaining work, in priority order

### 1. Compute dispatches

This is the most important gap. Most 3D games use compute.

- `RasterizerMetal::DispatchCompute` (`mtl_rasterizer.mm`) logs once and does nothing.
- Port `renderer_vulkan/vk_compute_pipeline.cpp` and the compute half of `vk_pipeline_cache.cpp`.
  - Add a `ComputePipelineCacheKey` (shader hash, shared memory size, workgroup size).
  - Run `EmitSPIRV`, then `TranslateSpirvToMsl` with the compute stage, then `newComputePipelineStateWithFunction:`.
  - Binding follows the same scheme as `GraphicsPipeline::BindStage`.
  - Set threadgroup memory with `setThreadgroupMemoryLength:`.
- Wire it into the disk cache. `PipelineCache::LoadDiskResources` currently skips compute entries (`load_compute`), and nothing serializes them.
- Check whether `mtl_shader_translator.cpp` needs compute-specific SPIRV-Cross options, such as the workgroup size from the SPIR-V.

### 2. Texture buffers and image buffers

- `GraphicsPipeline::BindStage` skips these and logs "Texture buffers are not implemented on Metal yet".
- Create `MTLTexture` views of buffers with `-[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]`. The offset and row pitch must meet `minimumLinearTextureAlignmentForPixelFormat:`.
- Fall back to a copy when the guest address isn't aligned.
- The buffer cache already tracks them: `BindGraphicsTextureBuffer` is called in `Configure`. The runtime side, `BufferCacheRuntime`, needs the view creation.

### 3. Draw paths that silently do nothing

- **`DrawIndirect`** isn't overridden, so indirect draws are dropped with **no log message**. Override it first and add a log, then implement it with `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`. Watch for draw counts read from guest memory: Metal has no multi-draw-indirect-count, so loop or use a compute pass.
- **`DrawTexture`** logs once and skips. Vulkan implements it with a blit helper.
- **Triangle fans, line loops and polygons:** `MaxwellToMTL::PrimitiveType` returns `nullopt`, and the draw is skipped with a warning. Rewrite them into triangle and line lists with a compute kernel, the same way quads are handled in `mtl_buffer_cache.mm` (`assemble_quads`).
- **Primitive restart** is always on in Metal. Indexed strips from games that disable it need their indices rewritten.

### 4. Queries and conditional rendering

`QueryCache` in `mtl_rasterizer.h` is a stub that writes results immediately, so occlusion queries always report the stub value. Implement it with a visibility result buffer (`setVisibilityResultMode:offset:`), and look at `video_core/query_cache` and Vulkan's `vk_query_cache.cpp`. `AccelerateConditionalRendering` stays false until then.

### 5. Texture cache gaps

- **MSAA copies:** `CopyImageMSAA` logs and skips, and `CanUploadMSAA` returns false. Use a draw that writes per sample, or resolve then copy.
- **Reinterpreting depth/stencil as color** (e.g. D24S8 ↔ R32): `CopyThroughBuffer` refuses formats that need conversion. Pack depth to the guest layout with the existing depth/stencil kernels (`mtl_texture_cache.mm`), then copy as raw bytes.
- **Resolution scaling:** `Image::IsRescaled` is always false, and `ScaleUp`/`ScaleDown` are no-ops. The shader side already pushes a `RescalingLayout` with a down factor of 1.
- **Missing formats:** `R32G32B32_FLOAT` and `G4R4` have no Metal texture format and are not copyable.

### 6. Geometry, tessellation and transform feedback

This is the correctness long tail, and the risk is high. Pipelines with tessellation or geometry stages fail on purpose and their draws are skipped. Shaders that need layer output through a geometry shader fail as well. Plan options are in the feature-gap table of `METAL-BACKEND-PLAN.md`:

- Geometry: a compute pre-pass or mesh shaders.
- Tessellation: a factor kernel plus post-tessellation vertex functions.
- Transform feedback: vertex-stage buffer writes.

### 7. Shader performance (optional)

- **Brand-new functions:** they take about 200 ms in Apple's compiler.
  - Measure `MTLCompileOptions` changes: `optimizationLevel = MTLLibraryOptimizationLevelSize`, and the math mode. Mind correctness under fast math; the depth kernels already use `precise::divide`.
  - Consider `MTLBinaryArchive` so compiled code doesn't depend on the system cache.
- **State-only variants:** pipelines that only differ in render state still wait about 16 ms for their pipeline state. These could be predicted or built earlier.
- **Draws that wait:** draws with ≤32 vertices wait for their pipeline even in async mode (`MAX_WAITING_DRAW_COUNT` in `mtl_pipeline_cache.mm`). One wait took 202 ms in testing.
- **Periodic flush:** the rasterizer flushes the command buffer every 2048 draws (`DRAWS_PER_FLUSH`). It was never tuned.

### 8. Frontend and packaging

- **MoltenVK still starts** when Metal is selected. The log shows a `Render.Vulkan` error about `VK_KHR_portability_enumeration`, likely from the Qt GPU enumeration. Skip it when the backend is Metal: `src/citron/CMakeLists.txt` and `startup_checks.cpp`, Phase 3 of the plan.
- **Build defaults:** Metal is off by default in the build script (see "Building with Metal" above).
- **iOS** has never been built or run with the Metal renderer.

## Traps already hit

- **dynarmic `ASSERT` is plain `assert()`.** Never put side effects in it, including in patches. `patches/dynarmic-arm64-emit-assert-side-effects.patch` fixes a hang this caused under `-DNDEBUG`.
- **MSL keywords** such as `vertex`, `fragment` and `kernel` can't be variable names in runtime shader strings. A local named `vertex` broke the index shader library once.
- **Metal blits only copy between identical pixel formats.** Packed 16-bit formats are named least-significant-bit first in Metal.
- **Depth and stencil copy separately** to and from buffers, using `MTLBlitOptionDepthFromDepthStencil` / `StencilFromDepthStencil`.
- **Fast math is on by default** in `newLibraryWithSource`. Unorm conversions need `precise::divide` plus `rint` to round-trip exactly.
- **ARC:** a reference to an `id` member needs `__strong`. Objective-C literals can't be `constexpr`.
- **Worker threads have no autorelease pool.** Wrap Metal work queued on `Common::ThreadWorker` in `@autoreleasepool`, as `GraphicsPipeline::Build` does.
- **Compiler settings:** citron builds with `-fno-rtti` and `-Werror` for unused, shadow and missing-declarations warnings. `video_core` adds `-Wno-shadow` on Apple.
- **New `.mm` files** must be added in two places in `src/video_core/CMakeLists.txt`: `target_sources` and the `set_source_files_properties` list that sets `LANGUAGE CXX` with `-x objective-c++ -fobjc-arc`. Tests go in the same two lists in `src/tests/CMakeLists.txt`.
- **Metal's Y axis points the other way:** SPIRV-Cross flips vertex Y with `flip_vert_y`, and viewports use Vulkan-style math with negative heights allowed. Vertex buffers sit at buffer index `30 - b`, and the translator keeps shader resources below them (`buffer_index_limit`).
