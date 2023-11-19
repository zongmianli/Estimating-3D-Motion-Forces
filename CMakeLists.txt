CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

# ----------------------------------------------------
# --- Set project info  ------------------------------
# ----------------------------------------------------
SET(PROJECT_NAME mfv)
SET(PROJECT_DESCRIPTION "Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video.")
SET(PROJECT_URL "http://github.com/zongmianli/human-object-contact")
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

SET(DOXYGEN_USE_MATHJAX YES)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- Load the Shared CMake submodule  ---------------
# ----------------------------------------------------
INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/apple.cmake)

list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots")
#MESSAGE(CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH})


IF(APPLE)
  APPLY_DEFAULT_APPLE_CONFIGURATION()
ENDIF(APPLE)

SETUP_PROJECT()
STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# ----------------------------------------------------
OPTION (EIGEN_NUMPY_ALIGNED "Directly aligned data between Numpy and Eigen" OFF)
OPTION (BUILD_UNIT_TESTS "Build the unitary tests" OFF)

IF(EIGEN_NUMPY_ALIGNED)
  ADD_DEFINITIONS(-DEIGENPY_ALIGNED)
ENDIF(EIGEN_NUMPY_ALIGNED)

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")

FINDPYTHON(3.8 EXACT REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
FIND_NUMPY()
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIRS})

# Add Boost path to include directories.
SET(BOOST_COMPONENTS python)
SEARCH_FOR_BOOST()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# Find Ceres solver
FIND_PACKAGE(Ceres REQUIRED)
#FIND_PACKAGE(Ceres REQUIRED PATHS "/home/zongmian/Predator/Software/install")
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

# Check for dependency using pkg-config
SET(PKG_CONFIG_PYWRAP_REQUIRES "eigenpy >= 1.3.1" "pinocchio >= 2.0.0")
FOREACH(dep ${PKG_CONFIG_PYWRAP_REQUIRES})
  ADD_COMPILE_DEPENDENCY(${dep})
ENDFOREACH(dep ${PKG_CONFIG_PYWRAP_REQUIRES})

MESSAGE(Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS})
MESSAGE(PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS})
MESSAGE(NUMPY_INCLUDE_DIRS: ${NUMPY_INCLUDE_DIRS})
MESSAGE(CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS})
MESSAGE(CERES_LIBRARIES: ${CERES_LIBRARIES})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
ADD_SUBDIRECTORY(solver)

# ----------------------------------------------------
# --- UNIT TESTS -------------------------------------
# ----------------------------------------------------
#ADD_SUBDIRECTORY(tests)

IF(EIGEN_NUMPY_ALIGNED)
  PKG_CONFIG_APPEND_CFLAGS("-DEIGENPY_ALIGNED")
ENDIF(EIGEN_NUMPY_ALIGNED)

PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_CFLAGS("-I${PYTHON_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_CFLAGS("-I${NUMPY_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_COMPONENTS})

SETUP_PROJECT_FINALIZE()
