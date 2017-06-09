#pragma once

#ifndef Q_MOC_RUN
// ros
#include <ros/ros.h>
#include <rviz/panel.h>
#include <pluginlib/class_list_macros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
//std
#include <stdio.h>
#include <map>
// qt
#include <QPainter>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QTableWidget>
#include <QComboBox>
#include <QTimer>
#include <QSlider>
// messages
#include <geometry_msgs/Pose.h>
#include <roboy_communication_middleware/MagneticSensor.h>
#include <visualization_msgs/Marker.h>
// common definitions
#include <common_utilities/CommonDefinitions.h>
#include <fstream>
#include <iostream>
#include <mutex>

#endif

using namespace std;
using namespace cv;

class BallInSocketPlugin : public rviz::Panel {
Q_OBJECT

public:
    BallInSocketPlugin(QWidget *parent = 0);

    ~BallInSocketPlugin();

    /**
     * Load all configuration data for this panel from the given Config object.
     * @param config rviz config file
     */
    virtual void load(const rviz::Config &config);

    /**
     * Save all configuration data from this panel to the given
     * Config object.  It is important here that you call save()
     * on the parent class so the class id and panel name get saved.
     * @param config rviz config file
     */
    virtual void save(rviz::Config config) const;
Q_SIGNALS:
    void newData();
public Q_SLOTS:
    void recordData(int state);
    void writeData();
public:
    void poseCallback(const geometry_msgs::PoseConstPtr& msg);
    void magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg);
private:
    ros::NodeHandle *nh;
    bool initialized = false, recording = false;
    pair<uint, uint> currentID;
    ros::AsyncSpinner *spinner;
    ros::Subscriber pose_sub, magnetic_sensor_sub;
    ofstream data_log;
    geometry_msgs::Pose pose;
    roboy_communication_middleware::MagneticSensor magneticSensors;
    mutex mux;
};