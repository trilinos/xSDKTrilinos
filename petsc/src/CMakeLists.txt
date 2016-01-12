#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

#
# src
#

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#
# Core Files
#

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  BelosPETScSolMgr.hpp
  Tpetra_PETScAIJGraph.hpp
  Tpetra_PETScAIJMatrix.hpp
  )

APPEND_SET(SOURCES
  BelosPETScSolMgr.cpp
  Tpetra_PETScAIJGraph.cpp
  Tpetra_PETScAIJMatrix.cpp
  )

#
# C) Define the targets for package's library/ies
#
TRIBITS_ADD_LIBRARY(
  xsdkpetsc
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )