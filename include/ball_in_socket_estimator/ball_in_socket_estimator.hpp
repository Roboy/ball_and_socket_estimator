#pragma once

#include <ros/ros.h>
#include <stdio.h>
#include <map>
#include <roboy_middleware_msgs/MagneticSensor.h>
#include <common_utilities/CommonDefinitions.h>
#include <common_utilities/rviz_visualization.hpp>
#include <fstream>
#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>
#include "grid3d.hpp"

using namespace std;
using namespace Eigen;

class BallInSocketEstimator : public rviz_visualization {

public:
    BallInSocketEstimator();

    ~BallInSocketEstimator();

    void magneticSensorCallback(const roboy_middleware_msgs::MagneticSensorConstPtr& msg);
private:
    ros::NodeHandlePtr nh;
    ros::Time start_time;
    boost::shared_ptr<ros::AsyncSpinner> spinner;
    ros::Subscriber magnetic_sensor_sub;
};
