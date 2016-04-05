TRIBITS_REPOSITORY_DEFINE_TPLS(
  MPI             "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/"  PT
  HYPRE           "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"    ST
  PETSC           "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"    ST
  TrilinosTpl     "cmake/tpls/"                                    ST
  )
