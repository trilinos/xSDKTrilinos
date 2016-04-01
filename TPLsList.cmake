TRIBITS_REPOSITORY_DEFINE_TPLS(
  MPI             "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/"  PT
  HYPRE           "${TRILINOS_SORUCE_DIR}/cmake/TPLs/"             ST
  PETSC           "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"    ST
  TrilinosTpl     "cmake/tpls/"                                    ST
  )
