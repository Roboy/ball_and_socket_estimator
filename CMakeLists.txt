CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ball_in_socket_estimator C CXX)

# Set target C++ standard and required compiler features
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE Debug FORCE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
# ros
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        geometry_msgs
        visualization_msgs
        roboy_middleware_msgs
        common_utilities
        tf
        tf_conversions
        pcl_ros
        )
catkin_package(CATKIN_DEPENDS message_runtime roboy_middleware_msgs )

#==============================================================================
# Dependencies
#==============================================================================
    # Eigen
    find_package(PkgConfig)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    include_directories(${EIGEN3_INCLUDE_DIRS})

    # libxml++
    find_package(TinyXML2 REQUIRED)
    include_directories(${LibXML++_INCLUDE_DIRS})

    find_package(PCL REQUIRED COMPONENTS common io visualization)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})

    find_package(CNPY REQUIRED)
    find_package(ZLIB REQUIRED)
    include_directories(${CNPY_INCLUDE_DIRS})

    # yaml-cpp
    pkg_check_modules( yaml-cpp REQUIRED yaml-cpp)
    include_directories(${yaml-cpp_INCLUDE_DIRS})

    include_directories(
            ${catkin_INCLUDE_DIRS}
            include/
    )
    #==============================================================================
    # Application
    #==============================================================================
    add_executable(ball_in_socket_estimator src/ball_in_socket_estimator.cpp src/pose_estimator.cpp)
    add_dependencies(ball_in_socket_estimator roboy_middleware_msgs_gencpp)
    target_link_libraries(ball_in_socket_estimator ${catkin_LIBRARIES} ${CNPY_LIBRARIES} ${ZLIB_LIBRARIES} ${PCL_LIBRARIES} ${yaml-cpp_LIBRARIES} )

    #==============================================================================
    # Installation
    #==============================================================================
    install(TARGETS
            ball_in_socket_estimator
            DESTINATION
            ${CATKIN_PACKAGE_LIB_DESTINATION}
            )
    install(FILES
            package.xml
            DESTINATION
            ${CATKIN_PACKAGE_SHARE_DESTINATION}
            )
