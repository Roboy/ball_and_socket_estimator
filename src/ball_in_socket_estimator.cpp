#include "ball_in_socket_estimator/ball_in_socket_estimator.hpp"

BallInSocketPlugin::BallInSocketPlugin(QWidget *parent)
        : rviz::Panel(parent) {
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketRvizPlugin",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    nh.reset(new ros::NodeHandle);

    spinner.reset(new ros::AsyncSpinner(3));

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
}

BallInSocketPlugin::~BallInSocketPlugin() {
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
        start_time = ros::Time::now();
    }else if (state == Qt::Unchecked) {
        recording = false;
        if(data_log.is_open())
            data_log.close();
    }
}

void BallInSocketPlugin::writeData(){
    std::lock_guard<std::mutex> lock(mux);
    Matrix3d rot = pose.block(0,0,3,3);
    Quaterniond q(rot);
    if(data_log.is_open()){
        data_log << (double)(ros::Time::now()-start_time).toSec() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                 << pose(0,3) << " " << pose(1,3) << " " << pose(2,3) << " ";
        for(uint i=0; i<magneticSensors.x.size();i++){
            data_log << magneticSensors.x[i] << " " << magneticSensors.y[i] << " " << magneticSensors.z[i] << " ";
        }
        data_log << endl;
    }
}

void BallInSocketPlugin::magneticSensorCallback(const roboy_communication_middleware::MagneticSensorConstPtr& msg){
    ROS_DEBUG_THROTTLE(10, "receiving magnetic data");
    magneticSensors = *msg;

    if(!getTransform("tracker_1","world",pose))
        ROS_WARN_THROTTLE(1,"could not get pose of tracker_1");

    for(uint i=0;i<magneticSensors.x.size();i++){
        Vector3d pos(i*0.5,0,1);
        Vector3d dir(magneticSensors.x[i]/1000.0f, magneticSensors.y[i]/1000.0f, magneticSensors.z[i]/1000.0f);
        dir = pose.block(0,0,3,3)*dir;
        pos = pos + dir;
        if(i==0)
            publishSphere(pos,"world","magneticSensors",message_id++,COLOR(1,0,0,1),0.01,0);
        else if(i==1)
            publishSphere(pos,"world","magneticSensors",message_id++,COLOR(0,1,0,1),0.01,0);
        else if(i==2)
            publishSphere(pos,"world","magneticSensors",message_id++,COLOR(0,0,1,1),0.01,0);
    }

    if(recording){
        emit newData();
    }
}

bool BallInSocketPlugin::getTransform(const char *from, const char *to, Matrix4d &transform){
    tf::StampedTransform trans;
    try {
        tf_listener.lookupTransform(to, from, ros::Time(0), trans);
    }
    catch (tf::TransformException ex) {
        ROS_WARN_THROTTLE(1, "%s", ex.what());
        return false;
    }

    Eigen::Affine3d trans_;
    tf::transformTFToEigen(trans, trans_);
    transform = trans_.matrix();
    return true;
}

PLUGINLIB_EXPORT_CLASS(BallInSocketPlugin, rviz::Panel)