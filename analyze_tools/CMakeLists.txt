project(analyze_tools)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

macro (add_sources)
    file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND source_files "${_relPath}/${_src}")
        else()
            list (APPEND source_files "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate source_files to parent directory
        set (source_files ${source_files} PARENT_SCOPE)
    endif()
endmacro()

find_package(ReaDDy REQUIRED)


add_subdirectory(src)

find_package(HDF5 COMPONENTS C HL REQUIRED)
find_package(Threads)
add_executable(analyze_tools_exec main_exec.cpp ${source_files})
target_include_directories(analyze_tools_exec PUBLIC libs/spdlog/include ${ReaDDy_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(analyze_tools_exec PUBLIC ${ReaDDy_LIBRARIES} ${HDF5_LIBRARIES} Threads::Threads)

add_subdirectory(libs/pybind11)
pybind11_add_module(analyze_tools main.cpp ${source_files})
target_include_directories(analyze_tools PRIVATE libs/spdlog/include ${ReaDDy_INCLUDE_DIRS})
target_link_libraries(analyze_tools PRIVATE ${ReaDDy_LIBRARIES})
