
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
SET(TRILINOS_SORUCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CACHE PATH "Path to Trilinos soruce tree")
SET(${PROJECT_NAME}_TRIBITS_DIR "${TRILINOS_SORUCE_DIR}/cmake/tribits"
  CACHE PATH "By default assume Teuchos is in Trilinos")
INCLUDE("${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake")
SET(TEUCHOS_STD_UNIT_TEST_MAIN
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Teuchos_StandardParallelUnitTestMain.cpp")
TRIBITS_PROJECT()


ENDIF(TRIBITS_PROCESSING_PACKAGE)
