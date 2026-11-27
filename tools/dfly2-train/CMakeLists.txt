# tools/dfly2-train - basalt-dfly2-train, the trainer of a DFly2 drafter, on LibTorch. Built with the root project
# when BASALT_DFLY2_TRAIN is on, or on its own:
#
#   cmake -S tools/dfly2-train -B build-dfly2-train -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
#         -DCMAKE_PREFIX_PATH=<libtorch>
#   cmake --build build-dfly2-train
#   ctest --test-dir build-dfly2-train -L cpu --output-on-failure
#
# LibTorch for Windows has no debug build: configure Release or RelWithDebInfo.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(basalt_dfly2_train CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
  endif()
  option(BUILD_TESTING "Build the trainer's tests" OFF)
  if(BUILD_TESTING)
    enable_testing()
  endif()
  # the root project's `warnings`, `platform` and `model_file` targets, from the one file both configurations include
  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/platform.cmake)
endif()

set(DFLY2_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

# LibTorch is not fetched: CMAKE_PREFIX_PATH or Torch_DIR names an unpacked release.
find_package(Torch REQUIRED)
separate_arguments(_dfly2_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# A test of the trainer: tests/<name>.cpp, linked to the libraries named after it, run with the arguments after ARGS,
# labelled `cpu`: none needs a GPU and none is shown one. Windows finds LibTorch's libraries through PATH, which the
# test's environment gets here.
function(dfly2_test name)
  cmake_parse_arguments(PARSE_ARGV 1 T "" "" "ARGS")
  add_executable(dfly2_${name} tests/${name}.cpp)
  target_link_libraries(dfly2_${name} PRIVATE ${T_UNPARSED_ARGUMENTS})
  add_test(NAME dfly2_${name} COMMAND dfly2_${name} ${T_ARGS})
  set_property(TEST dfly2_${name} APPEND PROPERTY LABELS cpu)
  set_property(TEST dfly2_${name} PROPERTY ENVIRONMENT_MODIFICATION "CUDA_VISIBLE_DEVICES=set:-1")
  if(WIN32)
    set_property(TEST dfly2_${name} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${TORCH_INSTALL_PREFIX}/lib")
  endif()
endfunction()

# The file formats, with no LibTorch in them: the safetensors checkpoints (the JSON reader is the model file's).
add_library(dfly2_io STATIC src/safetensors.cpp)
target_include_directories(dfly2_io PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(dfly2_io PUBLIC model_file)

# The drafter and what trains it.
add_library(dfly2_core STATIC src/sparse_attention.cpp src/drafter.cpp src/target_weights.cpp src/loss.cpp
  src/optimizer.cpp src/data.cpp src/checkpoint.cpp src/trainer.cpp)
target_compile_options(dfly2_core PUBLIC ${_dfly2_torch_flags})
if(MSVC)
  # LibTorch's headers instantiate standard-library templates with narrowing conversions and leave unreachable code
  # behind constant conditions; both are reported in the file that includes them.
  target_compile_options(dfly2_core PUBLIC /wd4267 /wd4702)
endif()
target_link_libraries(dfly2_core PUBLIC dfly2_io ${TORCH_LIBRARIES})

# The program. Its paths arrive on the command line, so it carries the engine's manifest: UTF-8 as the process code
# page and long paths. LibTorch's shared libraries are found at run time (PATH on Windows), not copied beside it.
add_executable(basalt-dfly2-train src/main.cpp)
target_link_libraries(basalt-dfly2-train PRIVATE dfly2_core)
basalt_ship_to_bin(basalt-dfly2-train)
if(MSVC)
  target_link_options(basalt-dfly2-train PRIVATE "/MANIFEST:EMBED"
                      "/MANIFESTINPUT:${DFLY2_ROOT}/src/platform/basalt.exe.manifest")
endif()

if(BUILD_TESTING)
  # the teacher shard's writer and reader (include/basalt/artifact/dfly2_shard.hpp)
  dfly2_test(shard_test platform)
  dfly2_test(safetensors_test dfly2_io)
  dfly2_test(attention_test dfly2_core)
  dfly2_test(drafter_test dfly2_core)
  dfly2_test(target_weights_test dfly2_core)
  dfly2_test(loss_test dfly2_core)
  dfly2_test(optimizer_test dfly2_core)
  # end to end on a tiny pack and shards, through the library and through the program
  dfly2_test(train_test dfly2_core ARGS $<TARGET_FILE:basalt-dfly2-train>)
endif()
