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
  Ifpack2_Hypre.hpp
  )

APPEND_SET(SOURCES
  Ifpack2_Hypre.cpp
  )

#
# C) Define the targets for package's library/ies
#
TRIBITS_ADD_LIBRARY(
  xsdkhypre
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )