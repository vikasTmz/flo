# Choose which chapters to compile
option(FLOW "Compile libigl tutorials chapter 1" OFF)

# Download tutorial data
#include(libigl_tutorial_data)

#if(NOT TARGET igl::glfw)
#    message(WARNING "Most tutorial executables depend on OpenGL and glfw. Use `cmake -DLIBIGL_GLFW=ON ..`")
#endif()

# Chapter 2
if(FLOW)
    igl_add_tutorial(801_ConformalWillmoreFlow igl::glfw)
    igl_add_tutorial(802_LaPlacian_Flow igl::glfw)
endif()