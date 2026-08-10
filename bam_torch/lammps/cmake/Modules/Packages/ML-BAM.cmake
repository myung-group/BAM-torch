cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")
