cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(betatrondiagnostics VERSION 0.0.0 LANGUAGES CXX)
find_package(Boost 1.66 REQUIRED)

add_library(betatrondiagnostics SHARED
  src/merit.cxx
  src/radiation.cxx
  src/trajectory.cxx
)

target_include_directories(betatrondiagnostics SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(betatrondiagnostics PRIVATE ${Boost_LIBRARIES})
set_target_properties(betatrondiagnostics PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(betatrondiagnostics PRIVATE -O3 -ffast-math)
install(TARGETS betatrondiagnostics DESTINATION lib)
