cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set (APP_NAME app-lbmMultiRes)
file(GLOB_RECURSE SrcFiles lbmMultiRes.cu)

add_executable(${APP_NAME} ${SrcFiles})

target_link_libraries(${APP_NAME} 
	PUBLIC libNeonSkeleton)

target_compile_definitions(${APP_NAME}
    PRIVATE MultiResLBMDir=${CMAKE_CURRENT_SOURCE_DIR}   
)
set_target_properties(${APP_NAME} PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${APP_NAME} PROPERTIES FOLDER "apps")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "${APP_NAME}" FILES ${SrcFiles})

add_test(NAME ${APP_NAME} COMMAND ${APP_NAME})