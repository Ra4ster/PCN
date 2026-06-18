cmake_minimum_required(VERSION 4.3.0)
project(pcn CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use modern package finding
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_executable(PCN src/main.cpp src/PCNLayer.cpp)

target_include_directories(PCN PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${BLAS_INCLUDE_DIRS}
)

target_link_libraries(PCN PRIVATE 
    ${BLAS_LIBRARIES} 
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(PCN PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(PCN PRIVATE -O3 -march=native -ffast-math)
endif()

set_target_properties(PCN PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)