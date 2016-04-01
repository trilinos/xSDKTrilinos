TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

SET(example_Belos_SOURCES Hypre_BelosEx.cpp)
SET(example_Solve_SOURCES Hypre_SolveEx.cpp)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Hypre_Belos_example
  SOURCES ${example_Belos_SOURCES}
  COMM serial mpi
  )
  
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Hypre_Solve_example
  SOURCES ${example_Solve_SOURCES}
  COMM serial mpi
  )