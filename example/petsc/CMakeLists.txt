#TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR})
#if (HAVE_PETSC)
#TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR/petsc})
#endif()
#if (HAVE_PETSC)
#TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR/hypre})
#endif()

#${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/petsc)




SET(example_Amesos2_SOURCES  Amesos2Test.cpp  )
SET(example_Anasazi_SOURCES  AnasaziTest.cpp  )
SET(example_TpetraPetsc_SOURCES  TpetraExt_petsc.cpp  )
SET(example_EpetraPETScEx_SOURCES EpetraPETScExFile.cpp)
SET(example_TpetraPETScEx_SOURCES TpetraPETScExFile.cpp)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_Amesos2
  SOURCES ${example_Amesos2_SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "Run has completed successfully!"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_Anasazi
  SOURCES ${example_Anasazi_SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "Run has completed successfully!"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_TpetraPetsc
  SOURCES ${example_TpetraPetsc_SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "Run has completed successfully!"
  ) 

if (HAVE_BELOS)
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_EpetraPETScEx
  SOURCES ${example_EpetraPETScEx_SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "Run has completed successfully!"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_TpetraPETScEx
  SOURCES ${example_TpetraPETScEx_SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "Run has completed successfully!"
  )
endif()
                            
