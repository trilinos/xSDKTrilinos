
SET(example_Amesos2_SOURCES PETSc_Amesos2Ex.cpp)
SET(example_Anasazi_SOURCES PETSc_AnasaziEx.cpp)
SET(example_Ifpack2_SOURCES PETSc_Ifpack2Ex.cpp)
SET(example_MueLu_SOURCES PETSc_MueLuEx.cpp)
SET(example_TpetraKSP_SOURCES Tpetra_KSPEx.cpp)
SET(example_EpetraKSP_SOURCES Epetra_KSPEx.cpp)


TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyxSDKTrilinosPetscExFiles
  SOURCE_FILES cage4.mtx
  )

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Amesos2)
IF (${PACKAGE_NAME}_ENABLE_Amesos2)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    PETSc_Amesos2_example
    SOURCES ${example_Amesos2_SOURCES}
    COMM serial mpi
    )
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Anasazi)
IF (${PACKAGE_NAME}_ENABLE_Anasazi)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    PETSc_Anasazi_example
    SOURCES ${example_Anasazi_SOURCES}
    COMM serial mpi
    )
ENDIF()
  
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  PETSc_Ifpack2_example
  SOURCES ${example_Ifpack2_SOURCES}
  COMM serial mpi
  )
  
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_MueLu)
IF (${PACKAGE_NAME}_ENABLE_MueLu)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    PETSc_MueLu_example
    SOURCES ${example_MueLu_SOURCES}
    COMM serial mpi
    )
ENDIF()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_TpetraKSP
  SOURCES ${example_TpetraKSP_SOURCES}
  COMM serial mpi
  ) 

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Epetra)
IF (${PACKAGE_NAME}_ENABLE_Epetra)
  ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EpetraExt)
  IF (${PACKAGE_NAME}_ENABLE_EpetraExt)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      example_EpetraKSP
      SOURCES ${example_EpetraKSP_SOURCES}
      COMM serial mpi
      )
  ENDIF()
ENDIF()
