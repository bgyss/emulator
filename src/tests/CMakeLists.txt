# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/fibers.cpp
    common/host_memory.cpp
    common/logging.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/unique_function.cpp
    common/uint128.cpp
    core/core_timing.cpp
    core/exit_callback.cpp
    core/internal_network/network.cpp
    precompiled_headers.h
    video_core/memory_tracker.cpp
    video_core/image_size.cpp
    input_common/calibration_configuration_job.cpp
)

# Test translation units are not unity-safe.
set_property(TARGET tests PROPERTY UNITY_BUILD OFF)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core input_common)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

if (CITRON_ENABLE_METAL)
    # Objective-C++ so the tests can drive Metal directly.
    target_sources(tests PRIVATE
        video_core/metal_scheduler.mm
        video_core/metal_shader_translator.mm
    )
    set_source_files_properties(
        video_core/metal_scheduler.mm
        video_core/metal_shader_translator.mm
        PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc"
        SKIP_PRECOMPILE_HEADERS ON
    )
    add_dependencies(tests host_shaders)
    target_include_directories(tests PRIVATE ${CMAKE_BINARY_DIR}/src/video_core/host_shaders/include)
    target_link_libraries(tests PRIVATE video_core "-framework Metal" "-framework Foundation")
endif()

add_test(NAME tests COMMAND tests)

if (CITRON_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(tests PRIVATE precompiled_headers.h)
endif()
