# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.1...3.23)

#set(CMAKE_CXX_STANDARD 17)

project( Curvature_Flow )

# LIBIGL
find_package( libigl REQUIRED )

if ( NOT LIBIGLL_FOUND )

  message(STATUS "This project requires the LIBIGL library, and will not be compiled.")
  return()  

endif()

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif()

# include for local directory

# include for local package


# Creating entries for all C++ files with "main" routine
# ##########################################################
add_executable(conformalwillmoreflow tutorial/801_ConformalWillmoreFlow/main.cpp) # you can change the excutable name
add_executable(laplacianflow tutorial/802_LaPlacian_Flow.cpp) # you can change the excutable name


#create_single_source_cgal_program( "mcfskel.cpp" )

