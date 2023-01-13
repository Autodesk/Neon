cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set (TUTORIAL_NAME tutorial-domainLevel)
file(GLOB_RECURSE SrcFiles domainLevel.cpp expandSphere.cu grad.cu)

add_executable(${TUTORIAL_NAME} ${SrcFiles})

target_link_libraries(${TUTORIAL_NAME} PUBLIC libNeonSkeleton)

set_target_properties(${TUTORIAL_NAME} PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(${TUTORIAL_NAME} PROPERTIES FOLDER "tutorials")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "${TUTORIAL_NAME}" FILES ${SrcFiles})
