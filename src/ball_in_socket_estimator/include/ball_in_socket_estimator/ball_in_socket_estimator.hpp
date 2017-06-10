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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#endif

using namespace std;
using namespace cv;
using namespace Eigen;

struct COLOR{
    COLOR(float r, float g, float b, float a):r(r),g(g),b(b),a(a){};
    float r,g,b,a;
};

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
    /**
     * Publishes a ray visualization marker
     * @param pos at this positon
     * @param dir direction
     * @param frame in this frame
     * @param message_id a unique id
     * @param ns namespace
     * @param rgda rgb color (0-1) plus transparancy
     * @param duration for this duration in seconds (0=forever)
     */
    void publishRay(Vector3d &pos, Vector3d &dir, const char* frame, const char* ns, int message_id, COLOR color, int duration=0);
public:
    void poseCallback(const geometry_msgs::PoseConstPtr& msg);
    void magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg);
private:
    void transformPublisher();
    ros::NodeHandle *nh;
    bool initialized = false, recording = false, publish_transform = true;
    pair<uint, uint> currentID;
    ros::AsyncSpinner *spinner;
    ros::Subscriber pose_sub, magnetic_sensor_sub;
    ros::Publisher visualization_pub;
    ofstream data_log;
    geometry_msgs::Pose pose;
    roboy_communication_middleware::MagneticSensor magneticSensors;
    mutex mux;
    COLOR colors[3] = {COLOR(255,0,0,1), COLOR(0,255,0,1), COLOR(0,0,255,1)};
    boost::shared_ptr<std::thread> transform_thread;
    tf::TransformBroadcaster tf_broadcaster;
    tf::Transform tf_camera;
};