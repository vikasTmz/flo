cmake_minimum_required(VERSION 3.1...3.23)
#cmake_minimum_required(VERSION 3.16)

project(Curvature_Flow)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
#igl_include(glfw)

# Add your project files
#file(GLOB SRC_FILES *.cpp)
#add_executable(${PROJECT_NAME} ${SRC_FILES})

add_executable(conformalwillmoreflow tutorial/801_ConformalWillmoreFlow/main.cpp) # you can change the excutable name
add_executable(laplacianflow tutorial/802_LaPlacian_Flow/main.cpp) # you can change the excutable name


#target_link_libraries(conformalwillmoreflow PUBLIC igl::glfw)
#target_link_libraries(laplacianflow PUBLIC igl::glfw)