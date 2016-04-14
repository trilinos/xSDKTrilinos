########################################################################
# See associated tribits/Copyright.txt file for copyright and license! #
########################################################################

SET(TPL_TrilinosTplsTpl_INCLUDE_DIRS "${Trilinos_TPL_INCLUDE_DIRS}"
  CACHE PATH "TrilinosTplsTpl include dirs")
SET(TPL_TrilinosTplsTpl_LIBRARIES "${Trilinos_TPL_LIBRARIES}"
  CACHE FILEPATH "TrilinosTplsTpl libraries")
SET(TPL_TrilinosTplsTpl_LIBRARY_DIRS "${Trilinos_TPL_LIBRARY_DIRS}"
  CACHE PATH "TrilinosTplsTpl library dirs")

LINK_DIRECTORIES(${TPL_TrilinosTplsTpl_LIBRARY_DIRS})

TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES( TrilinosTplsTpl
  REQUIRED_HEADERS dummy_header.h
  REQUIRED_LIBS_NAMES dummy_lib
  )
