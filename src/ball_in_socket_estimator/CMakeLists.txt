CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ball_in_socket_estimator C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE Debug FORCE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

# ros
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs geometry_msgs sensor_msgs cv_bridge image_transport visualization_msgs rviz roboy_communication_middleware)
catkin_package(CATKIN_DEPENDS message_runtime roboy_communication_middleware )
#==============================================================================
# Dependencies
#==============================================================================
    #Qt5
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt5 REQUIRED COMPONENTS Widgets Core Gui)
    include_directories(
            ${catkin_INCLUDE_DIRS}
            ${Qt5Widgets_INCLUDE_DIRS}
            ${Qt5Core_INCLUDE_DIRS}
            ${Qt5Gui_INCLUDE_DIRS}
            ${eigen3_INCLUDE_DIR}
    )

    find_package(Protobuf REQUIRED)

    # Eigen
    find_package(PkgConfig)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    include_directories(${EIGEN3_INCLUDE_DIRS})

    include_directories(
            ${catkin_INCLUDE_DIRS}
            include/
            ${CMAKE_CURRENT_SOURCE_DIR}/../communication/include
    )
    #==============================================================================
    # Application
    #==============================================================================
    qt5_wrap_cpp(MOC_FILES include/ball_in_socket_estimator/ball_in_socket_estimator.hpp)
    add_library(ball_in_socket_estimator src/ball_in_socket_estimator.cpp ${MOC_FILES})
    add_dependencies(ball_in_socket_estimator roboy_communication_middleware_gencpp)
    target_link_libraries(ball_in_socket_estimator ${QT_LIBRARIES} ${catkin_LIBRARIES})

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

