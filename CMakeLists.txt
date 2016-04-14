
IF (TRIBITS_PROCESSING_PACKAGE)

#
# Processing TriBITS package!
#

#
# A) Define the package
#

TRIBITS_PACKAGE(xSDKTrilinos)

#
# B) Set up package-specific options
#

# ToDo: Add options when needed

#
# C) Add the libraries, tests, and examples
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_PETSC)
IF (${PACKAGE_NAME}_ENABLE_PETSC)
  ADD_SUBDIRECTORY(petsc)
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_HYPRE)
IF (${PACKAGE_NAME}_ENABLE_HYPRE)
  ADD_SUBDIRECTORY(hypre)
ENDIF()

TRIBITS_EXCLUDE_AUTOTOOLS_FILES()

#
# D) Do standard postprocessing
#

TRIBITS_PACKAGE_POSTPROCESS()


ELSE(TRIBITS_PROCESSING_PACKAGE)


#
# Processing as a TriBITS project
#

# This CMakeLists.txt file is being processed as the TriBITS projects's base
# CMakeLists.txt file!  (See comments at bottom of this file.)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11 FATAL_ERROR)
INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/ProjectName.cmake")
PROJECT(${PROJECT_NAME} NONE)
SET(Trilinos_INSTALL_DIR ""  CACHE PATH "Path to base Trilinos installation")
IF ("${Trilinos_INSTALL_DIR}" STREQUAL "")
  MESSAGE(FATAL_ERROR "Must set Trilinos_INSTALL_DIR!")
ENDIF()
SET(CMAKE_PREFIX_PATH ${Trilinos_INSTALL_DIR} ${CMAKE_PREFIX_PATH})
FIND_PACKAGE(TriBITS REQUIRED)
SET(${PROJECT_NAME}_TRIBITS_DIR "${TRIBITS_DIR}"
  CACHE PATH "By default, use TriBITS in Trilinos install tree")
INCLUDE("${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake")
SET(TEUCHOS_STD_UNIT_TEST_MAIN
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Teuchos_StandardParallelUnitTestMain.cpp")
FIND_PACKAGE(Trilinos CONFIG REQUIRED)
# ToDo: Read in the compilers, linker, etc.
TRIBITS_PROJECT()


ENDIF(TRIBITS_PROCESSING_PACKAGE)
