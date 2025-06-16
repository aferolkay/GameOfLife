

# ————————————————————————————————
# 1. Host: require C++17
# ————————————————————————————————
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ————————————————————————————————
# 2. Device: force NVCC to use C++17
#    (manually append the right flag)
# ————————————————————————————————
# Make sure CMAKE_CUDA_FLAGS is defined
if(NOT DEFINED CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "")
endif()
# Append the C++17 switch for NVCC’s host compiler and its front‐end
#string(APPEND CMAKE_CUDA_FLAGS " --std=c++17 -Xcompiler=-std=c++17")
#string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=[sm_75,compute_75]")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75)
#set(CMAKE_CUDA_FLAGS
#  "-arch=compute_75 -ptx"
#)

# ————————————————————————————————
# 3. CUDA architecture + separable compilation
# ————————————————————————————————

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# ————————————————————————————————
# 4. Dependencies
# ————————————————————————————————
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLU REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${GLUT_INCLUDE_DIRS}
  )

# ————————————————————————————————
# 5. Sources
#      – .cpp in src/
#      – your .cu in src/
# ————————————————————————————————
message(STATUS "Project source directory: ${PROJECT_SOURCE_DIR}")
file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/LifeCuda.cu)

add_executable(Conway
  ${CPP_SOURCES}
  ${CUDA_SOURCES}
)

target_link_libraries(Conway PRIVATE
  OpenGL::OpenGL
  GLUT::GLUT
  GLU
)
