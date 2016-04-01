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
