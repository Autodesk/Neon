# ImGui core (always) + optional GLFW+OpenGL desktop backend.
#
# The ImGui *core* (imgui.cpp / draw / tables / widgets) is needed by both the
# desktop (GLFW+OpenGL) and the web (imgui-ws) backends, so it is built
# unconditionally. The desktop backend -- OpenGL, GLFW, and the
# imgui_impl_glfw / imgui_impl_opengl3 platform layers -- is compiled in only
# when CARBON_VIEWER_DESKTOP is ON. A web-only build (desktop OFF) therefore
# needs neither OpenGL nor GLFW dev packages, so headless hosts can serve the
# browser viewer (see CarbonViewerWs.cmake).
include(FetchContent)

FetchContent_Declare(
  carbon_imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.5
)
FetchContent_GetProperties(carbon_imgui)
if(NOT carbon_imgui_POPULATED)
  FetchContent_Populate(carbon_imgui)
endif()

set(CARBON_IMGUI_DIR "${carbon_imgui_SOURCE_DIR}")

add_library(carbon_imgui STATIC
  "${CARBON_IMGUI_DIR}/imgui.cpp"
  "${CARBON_IMGUI_DIR}/imgui_draw.cpp"
  "${CARBON_IMGUI_DIR}/imgui_tables.cpp"
  "${CARBON_IMGUI_DIR}/imgui_widgets.cpp")
add_library(carbon::imgui ALIAS carbon_imgui)

target_include_directories(carbon_imgui
  PUBLIC
    "${CARBON_IMGUI_DIR}"
    "${CARBON_IMGUI_DIR}/backends")
set_target_properties(carbon_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CARBON_VIEWER_DESKTOP)
  find_package(OpenGL REQUIRED)

  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    carbon_glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
  )
  FetchContent_MakeAvailable(carbon_glfw)

  target_sources(carbon_imgui
    PRIVATE
      "${CARBON_IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
      "${CARBON_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp")
  target_link_libraries(carbon_imgui PUBLIC glfw OpenGL::GL)
endif()
