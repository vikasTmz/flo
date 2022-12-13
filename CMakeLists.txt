cmake_minimum_required(VERSION 3.16)

project(Curvature_Flow)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

# Add your project files

add_executable(conformalwillmoreflow tutorial/801_ConformalWillmoreFlow/main.cpp) # you can change the excutable name
add_executable(laplacianflow tutorial/802_LaPlacian_Flow/main.cpp) # you can change the excutable name
add_executable(meshdecimate tutorial/803_decimate/main.cpp) # you can change the excutable name
# add_executable(conformalwillmoreflow_gui tutorial/804_CWFGUI/main.cpp) # you can change the excutable name


target_link_libraries(conformalwillmoreflow PUBLIC igl::glfw)
target_link_libraries(laplacianflow PUBLIC igl::glfw)
target_link_libraries(meshdecimate PUBLIC igl::glfw)
# target_link_libraries(conformalwillmoreflow_gui PUBLIC igl::glfw)