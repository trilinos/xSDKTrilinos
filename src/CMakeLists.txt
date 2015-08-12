#
#INCLUDE(TribitsLibraryMacros)

#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

#
# B) Define the header and source files (and directories)
#

SET(HEADERS "")
SET(SOURCES "")


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  )

if (HAVE_PETSC)
APPEND_SET(HEADERS Tpetra_PETScAIJMatrix.hpp)
APPEND_SET(SOURCES Tpetra_PETScAIJMatrix.cpp)
APPEND_SET(HEADERS Tpetra_PETScAIJGraph.hpp)
APPEND_SET(SOURCES Tpetra_PETScAIJGraph.cpp)
endif()
if (HAVE_HYPRE)
APPEND_SET(HEADERS Ifpack2_Hypre.hpp)
APPEND_SET(SOURCES Ifpack2_Hypre.cpp)
endif()
#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  xSDKTrilinos 
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )


