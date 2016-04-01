########################################################################
# See associated tribits/Copyright.txt file for copyright and license! #
########################################################################

MESSAGE("-- Using FIND_PACKAGE(Trilinos ...) ...")

SET(Triinos_INSTALL_DIR ""  CACHE PATH "Path to base Trilinos installation")

SET(CMAKE_PREFIX_PATH ${Trilinos_INSTALL_DIR} ${CMAKE_PREFIX_PATH})

FIND_PACKAGE(Trilinos CONFIG REQUIRED)

SET(TPL_TrilinosTpl_INCLUDE_DIRS "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}"
  CACHE PATH "TrilinosTpl include dirs")
SET(TPL_TrilinosTpl_LIBRARIES "${Trilinos_LIBRARIES};${Trilinos_TPL_LIBRARIES}"
  CACHE FILEPATH "TrilinosTpl libraries")
SET(TPL_TrilinosTpl_LIBRARY_DIRS "${Trilinos_LIBRARY_DIRS};${Trilinos_TPL_LIBRARY_DIRS}"
  CACHE PATH "TrilinosTpl library dirs")

LINK_DIRECTORIES(${TPL_TrilinosTpl_LIBRARY_DIRS})

TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES( TrilinosTpl
  REQUIRED_HEADERS dummy_header.h
  REQUIRED_LIBS_NAMES dummy_lib
  )
