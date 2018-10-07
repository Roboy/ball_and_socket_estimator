#pragma once

#ifndef Q_MOC_RUN
// ros
#include <ros/ros.h>
#include <rviz/panel.h>
#include <pluginlib/class_list_macros.h>
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
#include <roboy_communication_middleware/MagneticSensor.h>
#include <common_utilities/CommonDefinitions.h>
#include <common_utilities/rviz_visualization.hpp>
#include <fstream>
#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>

#endif

using namespace std;
using namespace Eigen;

class BallInSocketPlugin : public rviz::Panel, rviz_visualization {
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
    void magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg);
private:
    bool getTransform(const char *from, const char *to, Matrix4d &transform);
    ros::NodeHandlePtr nh;
    ros::Time start_time;
    bool initialized = false, recording = false, publish_transform = true;
    pair<uint, uint> currentID;
    boost::shared_ptr<ros::AsyncSpinner> spinner;
    ros::Subscriber magnetic_sensor_sub;
    ros::Publisher visualization_pub, magnetic_field_pub;
    ofstream data_log;
    roboy_communication_middleware::MagneticSensor magneticSensors;
    Matrix4d pose;
    int message_id = 0;
    mutex mux;
    boost::shared_ptr<std::thread> transform_thread;
    tf::TransformBroadcaster tf_broadcaster;
    tf::TransformListener tf_listener;
    tf::Transform tf_camera;
    pcl::PointCloud<pcl::PointXYZRGB> magnetic_field;
};