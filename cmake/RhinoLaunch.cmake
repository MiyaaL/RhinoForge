# An approved rpu_ops checkout carries the matching public Launch headers and
# shared library without a CMake package. This explicit mode never searches a
# different installed SDK. Otherwise preserve the release package contract.
set(RHINO_LAUNCH_DIR "" CACHE PATH "Launch root from an approved rpu_ops/runtime/launch")
if(RHINO_LAUNCH_DIR)
    get_filename_component(RHINO_LAUNCH_DIR "${RHINO_LAUNCH_DIR}" ABSOLUTE)
    set(_rhino_release "${RHINO_LAUNCH_DIR}/../RELEASE.txt")
    if(NOT EXISTS "${_rhino_release}")
        message(FATAL_ERROR "RHINO_LAUNCH_DIR requires adjacent runtime/RELEASE.txt: ${_rhino_release}")
    endif()
    file(STRINGS "${_rhino_release}" _rhino_version REGEX "^launch_version=")
    if(NOT _rhino_version STREQUAL "launch_version=1.0.0")
        message(FATAL_ERROR "RHINO_LAUNCH_DIR requires launch_version=1.0.0 in ${_rhino_release}")
    endif()
    foreach(_rhino_header IN ITEMS
            rhino_launch_def.h rhino_launch_features.h rhino_launch_program.h
            rhino_launch_kernel.h rhino_launch_buffer.h rhino_launch_queue.h version.h)
        if(NOT EXISTS "${RHINO_LAUNCH_DIR}/include/${_rhino_header}")
            message(FATAL_ERROR "RHINO_LAUNCH_DIR is missing public header: ${_rhino_header}")
        endif()
    endforeach()
    # Clear a previous configure's cached result if the selected root changed.
    unset(_rhino_library CACHE)
    unset(_rhino_library)
    find_library(_rhino_library NAMES librhino_launch.so
        PATHS "${RHINO_LAUNCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
    add_library(rhino_launch::rhino_launch SHARED IMPORTED)
    set_target_properties(rhino_launch::rhino_launch PROPERTIES
        IMPORTED_LOCATION "${_rhino_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${RHINO_LAUNCH_DIR}/include")
    message(STATUS "Rhino Launch 1.0.0 (repository runtime): ${_rhino_library}")
else()
    find_package(rhino_launch 1.0.0 EXACT CONFIG REQUIRED)
endif()
