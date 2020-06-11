CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ball_in_socket_estimator C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE Debug FORCE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
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


find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
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

    include_directories(
            ${catkin_INCLUDE_DIRS}
            include/
    )
    #==============================================================================
    # Application
    #==============================================================================
    add_executable(ball_in_socket_estimator src/ball_in_socket_estimator.cpp)
    add_dependencies(ball_in_socket_estimator roboy_middleware_msgs_gencpp)
    target_link_libraries(ball_in_socket_estimator ${catkin_LIBRARIES})

    add_executable(magnetic_field_visualization src/magnetic_field_visualization.cpp)
    target_link_libraries(magnetic_field_visualization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${TinyXML2_LIBRARIES})

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
