include_directories(${Conway_SOURCE_DIR}/include)
include_directories(${GLUT_INCLUDE_DIRS} )

set(CUDA_SOURCES ${Conway_SOURCE_DIR}/src/Life.cu)
set(CUDA_HEADERS ${Conway_SOURCE_DIR}/include/Life.cuh)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_75")
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
cuda_add_library(afer ${CUDA_SOURCES} ${CUDA_HEADERS})


file(GLOB SOURCES "*.cpp")
add_executable(Conway ${SOURCES})
target_link_libraries(Conway OpenGL::OpenGL GLUT::GLUT GLU afer)
