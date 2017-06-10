#include "ball_in_socket_estimator/ball_in_socket_estimator.hpp"

BallInSocketPlugin::BallInSocketPlugin(QWidget *parent)
        : rviz::Panel(parent) {
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketRvizPlugin",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    nh = new ros::NodeHandle;

    spinner = new ros::AsyncSpinner(3);

    pose_sub = nh->subscribe("/mocap/MarkerPose", 100, &BallInSocketPlugin::poseCallback, this);
    magnetic_sensor_sub = nh->subscribe("/roboy/middleware/MagneticSensor", 100, &BallInSocketPlugin::magneticSensorCallback, this);
    visualization_pub = nh->advertise<visualization_msgs::Marker>("/visualization_marker", 1);

    QObject::connect(this, SIGNAL(newData()), this, SLOT(writeData()));

    // Create the main layout
    QHBoxLayout *mainLayout = new QHBoxLayout;

    // Create the frame to hold all the widgets
    QFrame *mainFrame = new QFrame();

    QCheckBox *record = new QCheckBox(tr("record"));
    connect(record, SIGNAL(stateChanged(int)), this, SLOT(recordData(int)));
    mainLayout->addWidget(record);

    // Add the frame to the main layout
    mainLayout->addWidget(mainFrame);

    // Remove margins to reduce space
    mainLayout->setContentsMargins(0, 0, 0, 0);

    this->setLayout(mainLayout);

    tf_camera.setOrigin(tf::Vector3(0, 0, 1.0));
    tf::Quaternion quat;
    quat.setRPY(M_PI / 2, 0, M_PI / 2);
    tf_camera.setRotation(quat);

    publish_transform = true;
    if(transform_thread==nullptr){
        transform_thread = boost::shared_ptr<std::thread>(new std::thread(&BallInSocketPlugin::transformPublisher, this));
        transform_thread->detach();
    }
}

BallInSocketPlugin::~BallInSocketPlugin() {
    delete nh;
    delete spinner;
}

void BallInSocketPlugin::load(const rviz::Config &config) {
    rviz::Panel::load(config);
}

void BallInSocketPlugin::save(rviz::Config config) const {
    rviz::Panel::save(config);
}

void BallInSocketPlugin::recordData(int state){
    if (state == Qt::Checked) {
        data_log = ofstream("data.log");
        data_log << "time q.x q.y q.z q.w x y z magnet.x magnet.y magnet.z magnet.x magnet.y magnet.z magnet.x magnet.y magnet.z\n";
        recording = true;
    }else if (state == Qt::Unchecked) {
        recording = false;
        if(data_log.is_open())
            data_log.close();
    }
}

void BallInSocketPlugin::writeData(){
    std::lock_guard<std::mutex> lock(mux);
    if(data_log.is_open()){
        data_log << ros::Time::now().nsec << " " << pose.orientation.x << " " << pose.orientation.y << " " << pose.orientation.z << " " << pose.orientation.w << " "
                 << pose.position.x << " " << pose.position.y << " " << pose.position.z << " ";
        for(uint i=0; i<magneticSensors.x.size();i++){
            data_log << magneticSensors.x[i] << " " << magneticSensors.y[i] << " " << magneticSensors.z[i] << " ";
        }
        data_log << endl;
    }
}

void BallInSocketPlugin::poseCallback(const geometry_msgs::PoseConstPtr &msg) {
    ROS_INFO_THROTTLE(1, "receiving pose data");
    if(recording){
        emit newData();
    }
    pose = *msg;
    visualization_msgs::Marker msg2;
    msg2.header.frame_id = "world";
    msg2.ns = "markerModel";
    msg2.type = visualization_msgs::Marker::MESH_RESOURCE;
    msg2.color.r = 0.0f;
    msg2.color.g = 1.0f;
    msg2.color.b = 0.0f;
    msg2.color.a = 0.5;
    msg2.scale.x = 1.0;
    msg2.scale.y = 1.0;
    msg2.scale.z = 1.0;
    msg2.lifetime = ros::Duration();
    msg2.action = visualization_msgs::Marker::ADD;
    msg2.header.stamp = ros::Time::now();
    msg2.id = 81;
    msg2.pose.position.x = 0;
    msg2.pose.position.y = 0;
    msg2.pose.position.z = 0;
    msg2.pose.orientation = msg->orientation;
    msg2.mesh_resource = "package://tracking_node/models/markermodel.STL";
    visualization_pub.publish(msg2);
}

void BallInSocketPlugin::magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg){
    ROS_INFO_THROTTLE(1, "receiving magnetic data");
//    if(recording){
//        emit newData();
//    }
    magneticSensors = *msg;

    for(uint i=0;i<magneticSensors.x.size();i++){
        Vector3d pos(0,0,0);
        Vector3d dir(magneticSensors.x[i]/100.0f, magneticSensors.y[i]/100.0f, magneticSensors.z[i]/100.0f);
        publishRay(pos,dir,"world","magneticSensors",i,colors[i],1);
    }
}

void BallInSocketPlugin::publishRay(Vector3d &pos, Vector3d &dir, const char *frame, const char *ns, int message_id, COLOR color, int duration) {
    visualization_msgs::Marker arrow;
    arrow.ns = ns;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.color.r = color.r;
    arrow.color.g = color.g;
    arrow.color.b = color.b;
    arrow.color.a = color.a;
    arrow.lifetime = ros::Duration(duration);
    arrow.scale.x = 0.003;
    arrow.scale.y = 0.03;
    arrow.scale.z = 0.03;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.header.stamp = ros::Time::now();
    arrow.header.frame_id = frame;
    arrow.id = message_id;
    arrow.points.clear();
    geometry_msgs::Point p;
    p.x = pos(0);
    p.y = pos(1);
    p.z = pos(2);
    arrow.points.push_back(p);
    p.x += dir(0);
    p.y += dir(1);
    p.z += dir(2);
    arrow.points.push_back(p);
    visualization_pub.publish(arrow);
}

void BallInSocketPlugin::transformPublisher(){
    ros::Rate rate(5);
    while(publish_transform){
        tf_broadcaster.sendTransform(tf::StampedTransform(tf_camera, ros::Time::now(), "map", "world"));
        rate.sleep();
    }
}

PLUGINLIB_EXPORT_CLASS(BallInSocketPlugin, rviz::Panel)