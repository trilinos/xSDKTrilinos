
SET(example_Amesos2_SOURCES PETSc_Amesos2Ex.cpp)
SET(example_Anasazi_SOURCES PETSc_AnasaziEx.cpp)
SET(example_Ifpack2_SOURCES PETSc_Ifpack2Ex.cpp)
SET(example_MueLu_SOURCES PETSc_MueLuEx.cpp)
SET(example_TpetraKSP_SOURCES Tpetra_KSPEx.cpp)
SET(example_EpetraKSP_SOURCES Epetra_KSPEx.cpp)


TRIBITS_ADD_EXECUTABLE(
  PETSc_Amesos2_example
  SOURCES ${example_Amesos2_SOURCES}
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  PETSc_Anasazi_example
  SOURCES ${example_Anasazi_SOURCES}
  COMM serial mpi
  )
  
TRIBITS_ADD_EXECUTABLE(
  PETSc_Ifpack2_example
  SOURCES ${example_Ifpack2_SOURCES}
  COMM serial mpi
  )
  
TRIBITS_ADD_EXECUTABLE(
  PETSc_MueLu_example
  SOURCES ${example_MueLu_SOURCES}
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  example_TpetraKSP
  SOURCES ${example_TpetraKSP_SOURCES}
  COMM serial mpi
  ) 

TRIBITS_ADD_EXECUTABLE(
  example_EpetraKSP
  SOURCES ${example_EpetraKSP_SOURCES}
  COMM serial mpi
  )
                            
