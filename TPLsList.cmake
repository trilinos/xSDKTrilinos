TRIBITS_REPOSITORY_DEFINE_TPLS(
  MPI             "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/"  PT
  TrilinosTplsTpl "cmake/tpls/"                                    ST
  HYPRE           "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"    ST
  TrilinosPkgsTpl "cmake/tpls/"                                    ST
  PETSC           "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"    ST
  )

# This ordering of TPLs is worth explaining.  First, the MPI TPL is needed so
# TriBITS will provide TPL support.
#
# Next comes the dummy TPL TrilinosTplsTpl.  This TPL does nothing but grab
# Trilinos_TPL_INCLUDE_DIRS and Trilinos_TPL_LIBRARIES provided by the
# TrilinosConfig.cmake and place it on the compile and link lines before the
# HYPRE and other TPLs.  This is because this will contain the include dirs
# and libraries for BLAS, LAPACK, METIS, ParMETIS, and SuperLUDist.  At least
# BLAS and LAPACK get linked into HYPRE.
#
# Next comes HYPRE which is directly used by xSDKTrilinos
#
# Next comes the dummy TPL TrilinosPkgsTpl.  This TPL does nothing but grab
# Trilinos_INCLUDE_DIRS and Trilinos_LIBRARIES provided by the
# TrilinosConfig.cmake and place it on the compile and link lines after HYPRE
# but before PETSC.  This is placed before PETSC in case PETSC was built with
# support for Trilinos.  In this case, the Trilinos libraries need to be
# listed before the PETSC library.
#
# Last comes the PETSC TPL.  This provides includes and libs needed by
# xSDKTrilinos.

