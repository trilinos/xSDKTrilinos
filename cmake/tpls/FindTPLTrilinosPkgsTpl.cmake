########################################################################
# See associated tribits/Copyright.txt file for copyright and license! #
########################################################################

SET(TPL_TrilinosPkgsTpl_INCLUDE_DIRS "${Trilinos_INCLUDE_DIRS}"
  CACHE PATH "TrilinosPkgsTpl include dirs")
SET(TPL_TrilinosPkgsTpl_LIBRARIES "${Trilinos_LIBRARIES}"
  CACHE FILEPATH "TrilinosPkgsTpl libraries")
SET(TPL_TrilinosPkgsTpl_LIBRARY_DIRS "${Trilinos_LIBRARY_DIRS}"
  CACHE PATH "TrilinosPkgsTpl library dirs")

LINK_DIRECTORIES(${TPL_TrilinosPkgsTpl_LIBRARY_DIRS})

TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES( TrilinosPkgsTpl
  REQUIRED_HEADERS dummy_header.h
  REQUIRED_LIBS_NAMES dummy_lib
  )
