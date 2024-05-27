# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
find_package(Boost COMPONENTS system thread filesystem)
if (Boost_FOUND)
  message("Find Boost in system")
  find_package(Boost REQUIRED COMPONENTS system thread filesystem)
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})
else()
  hunter_add_package(Boost COMPONENTS system thread filesystem)
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS Boost::system Boost::filesystem Boost::thread)
endif ()


# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
find_package(GLOG)
if (GLOG_FOUND)
  message("Find GLOG in system")
  include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})
else()
  hunter_add_package(glog)
  find_package(glog CONFIG REQUIRED)
  list(APPEND Caffe_LINKER_LIBS glog::glog)
endif ()

# ---[ Google-gflags
find_package(GFlags)
if (gflags_FOUND)
  message("Find gflags in system")
  include_directories(${GFLAGS_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})
else()
  hunter_add_package(gflags)
  find_package(gflags CONFIG REQUIRED)
  list(APPEND Caffe_LINKER_LIBS gflags)
endif ()


# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

## ---[ HDF5
#if (USE_HDF5)
#  hunter_add_package(hdf5)
#  find_package(ZLIB CONFIG REQUIRED)
#  find_package(szip CONFIG REQUIRED)
#  find_package(hdf5 CONFIG REQUIRED)
#  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
#  list(APPEND Caffe_LINKER_LIBS hdf5 hdf5_hl)
#  add_definitions(-DUSE_HDF5)
#endif ()

# ---[ HDF5
if (USE_HDF5)
  find_package(HDF5 COMPONENTS HL REQUIRED)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  add_definitions(-DUSE_HDF5)
endif ()
# ---[ LMDB
if(USE_LMDB)
  hunter_add_package(lmdb)
  find_package(liblmdb CONFIG REQUIRED)
  list(APPEND Caffe_LINKER_LIBS liblmdb::lmdb)
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  hunter_add_package(crc32c)
  find_package(Crc32c CONFIG REQUIRED)
  hunter_add_package(leveldb)
  find_package(leveldb CONFIG REQUIRED)
  list(APPEND Caffe_LINKER_LIBS leveldb::leveldb)
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  hunter_add_package(sleef)
  find_package(sleef CONFIG REQUIRED)
  list(APPEND Caffe_LINKER_LIBS sleef::sleef)
endif()

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
  endif()

  list(APPEND Caffe_DEFINITIONS PUBLIC -DCPU_ONLY)
endif()

if(USE_NCCL)
  find_package(NCCL REQUIRED)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARIES})
  add_definitions(-DUSE_NCCL)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV)
  if (OpenCV_FOUND)
    if(OpenCV_VERSION MATCHES "^2\\.")
      find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
    else()
      find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
    endif()
    message("Found OpenCV in system(${OpenCV_CONFIG_PATH})")
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
  else()
    hunter_add_package(OpenCV)
    if(OpenCV_VERSION MATCHES "^2\\.")
      find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui imgproc)
    else()
      find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui imgproc imgcodecs)
    endif()
    list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
  endif ()
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_OPENCV)
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Open" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS)
    if (OpenBLAS_FOUND)
      message("Find OpenBLAS in system, include_dir: " ${OpenBLAS_INCLUDE_DIR})
      include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
      list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})
    else()
      hunter_add_package(OpenBLAS)
      find_package(OpenBLAS CONFIG REQUIRED)
      list(APPEND Caffe_LINKER_LIBS OpenBLAS::OpenBLAS)
    endif ()
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND Caffe_DEFINITIONS -DUSE_ACCELERATE)
    endif()
  endif()
endif()

# ---[ Python
if(BUILD_python)
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

      STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    # disable Python 3 search
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
  endif()
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(BUILD_python_layer)
      add_definitions(-DWITH_PYTHON_LAYER)
      include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
      list(APPEND Caffe_LINKER_LIBS ${PYTHON_LIBRARIES} Boost::system Boost::filesystem Boost::thread)
    endif()
  endif()
endif()

# ---[ Matlab
if(BUILD_matlab)
  find_package(MatlabMex)
  if(MATLABMEX_FOUND)
    set(HAVE_MATLAB TRUE)
  endif()

  # sudo apt-get install liboctave-dev
  find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

  if(HAVE_MATLAB AND Octave_compiler)
    set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
    set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
  endif()
endif()

# ---[ Doxygen
if(BUILD_docs)
  find_package(Doxygen)
endif()
