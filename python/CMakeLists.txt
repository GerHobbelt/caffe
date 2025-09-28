if(NOT HAVE_PYTHON)
  message(STATUS "Python interface is disabled or not all required dependencies found. Building without it...")
  return()
endif()

file(GLOB_RECURSE python_srcs ${PROJECT_SOURCE_DIR}/python/*.cpp)

add_library(pycaffe SHARED ${python_srcs})
caffe_default_properties(pycaffe)
set_target_properties(pycaffe PROPERTIES PREFIX "" OUTPUT_NAME "_caffe")
target_include_directories(pycaffe PUBLIC ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(pycaffe PUBLIC ${Caffe_LINK} ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARY})

# ---[ Install
# scripts
file(GLOB python_files *.py requirements.txt)
install(FILES ${python_files} DESTINATION lib/python3.11/dist-packages)

# module
install(DIRECTORY caffe
    DESTINATION lib/python3.11/dist-packages
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "ilsvrc_2012_mean.npy"
    PATTERN "test" EXCLUDE
    )

# _caffe.so
install(TARGETS pycaffe DESTINATION lib/python3.11/dist-packages/caffe)

if(UNIX OR APPLE)
    add_custom_target(copy_files ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory /usr/local/lib/python3.11/dist-packages/caffe
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/lib/_caffe.so /usr/local/lib/python3.11/dist-packages/caffe/
        COMMENT "Copying files to /usr/local/lib/python3.11/dist-packages"
        DEPENDS pycaffe
    )
endif()

