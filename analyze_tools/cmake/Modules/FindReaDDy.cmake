# - Try to find ReaDDy
# Once done this will define
#  ReaDDy_FOUND - System has ReaDDy
#  ReaDDy_INCLUDE_DIRS - The ReaDDy include directories
#  ReaDDy_LIBRARIES - The libraries needed to use ReaDDy

find_path(ReaDDy_INCLUDE_DIR readdy/readdy.h HINTS PATH_SUFFIXES readdy)

find_library(ReaDDy_LIBRARY NAMES readdy)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ReaDDy_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ReaDDy DEFAULT_MSG ReaDDy_LIBRARY ReaDDy_INCLUDE_DIR)

mark_as_advanced(ReaDDy_INCLUDE_DIR ReaDDy_LIBRARY )

set(ReaDDy_LIBRARIES ${ReaDDy_LIBRARY} )
set(ReaDDy_INCLUDE_DIRS ${ReaDDy_INCLUDE_DIR} )