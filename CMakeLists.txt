CMAKE_MINIMUM_REQUIRED( VERSION 3.21.1 )

ADD_DEFINITIONS( -std=c++14 )

SET( EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" )

#find required eigen
FIND_PACKAGE( Eigen3 )

#directory of eigen headers
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )

#find required opencv
FIND_PACKAGE( OpenCV REQUIRED )

#directory of opencv headers
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY( ${PROJECT_SOURCE_DIR}/src src_file )
AUX_SOURCE_DIRECTORY( ${PROJECT_SOURCE_DIR} src_file )

INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )
INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include/dbow3" )

#directory of opencv link libs
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})

LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/lib/")
MESSAGE( STATUS "lib include path: " ${PROJECT_SOURCE_DIR}/lib/ )

FILE(
    GLOB_RECURSE mains RELATIVE "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src/training.cpp"
    "${PROJECT_SOURCE_DIR}/src/loop_closure_detect.cpp"
)

FOREACH(mainfile IN LISTS mains)
	GET_FILENAME_COMPONENT(mainname ${mainfile} NAME_WE)
	ADD_EXECUTABLE(${mainname} ${mainfile} ${src_test})

	TARGET_LINK_LIBRARIES( ${mainname} ${OpenCV_LIBS} ) 

	TARGET_LINK_LIBRARIES( ${mainname} libDBoW3.so )

ENDFOREACH()

