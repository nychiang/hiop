
#[[

Looks for `umfpack` library and header directory.

Exports target `UMFPACK` which links to umfpack.(so|a)
and add include directories where umfpack.h was found.

Users may set the following variables:

- HIOP_UMFPACK_DIR

]]

find_library(UMFPACK_LIBRARY
  NAMES
  umfpack
  PATHS
  ${UMFPACK_DIR} $ENV{UMFPACK_DIR} ${HIOP_UMFPACK_DIR}
  PATH_SUFFIXES
  lib)

find_path(UMFPACK_INCLUDE_DIR
  NAMES
  umfpack.h
  PATHS
  ${UMFPACK_DIR} $ENV{UMFPACK_DIR} ${HIOP_UMFPACK_DIR}
  PATH_SUFFIXES
  include
  suitesparse
  ufsparse)

set(UMFPACK_INCLUDE_DIR "${UMFPACK_INCLUDE_DIR}" CACHE PATH "Path to umfpack.h")
set(UMFPACK_LIBRARY "${UMFPACK_LIBRARY}" CACHE PATH "Path to umfpack library")

if(UMFPACK_LIBRARY)
  message(STATUS "Found umfpack include: ${UMFPACK_INCLUDE_DIR}")
  message(STATUS "Found umfpack library: ${UMFPACK_LIBRARY}")
  add_library(UMFPACK INTERFACE)
  target_link_libraries(UMFPACK INTERFACE ${UMFPACK_LIBRARY})
  target_include_directories(UMFPACK INTERFACE ${UMFPACK_INCLUDE_DIR})
else()
  message(STATUS "UMFPACK was not found.")
endif()

