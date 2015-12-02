INCLUDE(TribitsPackageMacros)
INCLUDE(TribitsAddOptionAndDefine)

#
# A) Define the package
#

TRIBITS_PACKAGE(xSDKTrilinos)

#
# B) Set up package-specific options
#

TRIBITS_ADD_OPTION_AND_DEFINE(${PACKAGE_NAME}_EXPLICIT_TEMPLATE_INSTANTIATION
  PHX_ETI
  "Build with explicit template instation enabled."
  OFF )

TRIBITS_ADD_OPTION_AND_DEFINE(${PACKAGE_NAME}_USING_PETSC
  HAVE_PETSC
  "Enable Tpetra/PETSc adapters"
  OFF )

TRIBITS_ADD_OPTION_AND_DEFINE(${PACKAGE_NAME}_USING_HYPRE
  HAVE_HYPRE
  "Enable Tpetra/HYPRE adapters"
  OFF )


TRIBITS_ADD_DEBUG_OPTION()

#
# C) Add the libraries, tests, and examples
#

ADD_SUBDIRECTORY(src)

TRIBITS_ADD_TEST_DIRECTORIES(test)

TRIBITS_ADD_EXAMPLE_DIRECTORIES(example)


TRIBITS_EXCLUDE_AUTOTOOLS_FILES()
#
# D) Do standard postprocessing
#

TRIBITS_PACKAGE_POSTPROCESS()
