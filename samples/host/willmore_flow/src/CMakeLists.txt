# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.1...3.23)

set(CMAKE_CXX_STANDARD 17)

project( Surface_mesh_skeletonization )


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
#add_executable(openmp_test openmp_test.cpp) # you can change the excutable name
#add_executable(main_v2 main_v2.cpp) # you can change the excutable name


create_single_source_cgal_program( "main_v2.cpp" )

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main_v2 PUBLIC OpenMP::OpenMP_CXX)
endif()

