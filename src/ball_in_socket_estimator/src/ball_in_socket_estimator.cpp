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
        data_log << ros::Time::now().nsec << " " << pose.orientation.x << " " << pose.orientation.y << " " << pose.orientation.z << " "
                << pose.orientation.w << " " << pose.position.x << " " << pose.position.y << " " << pose.position.z << " ";
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
}

void BallInSocketPlugin::magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg){
    ROS_INFO_THROTTLE(1, "receiving magnetic data");
//    if(recording){
//        emit newData();
//    }
    magneticSensors = *msg;
}

PLUGINLIB_EXPORT_CLASS(BallInSocketPlugin, rviz::Panel)