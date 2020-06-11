#include "ball_in_socket_estimator/ball_in_socket_estimator.hpp"

BallInSocketEstimator::BallInSocketEstimator(){
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketEstimator",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    nh.reset(new ros::NodeHandle);
    spinner.reset(new ros::AsyncSpinner(0));

    magnetic_sensor_sub = nh->subscribe("/roboy/middleware/MagneticSensor", 100, &BallInSocketEstimator::magneticSensorCallback, this);
    visualization_pub = nh->advertise<visualization_msgs::Marker>("/visualization_marker", 1);
}

BallInSocketEstimator::~BallInSocketEstimator() {
}


void BallInSocketEstimator::magneticSensorCallback(const roboy_middleware_msgs::MagneticSensorConstPtr& msg){
    ROS_DEBUG_THROTTLE(10, "receiving magnetic data");

}


int main(){
    Grid<float> grid;
    for (unsigned i = 0; i < 1000; ++i) {
        // create a random location
        Vec3f result = grid.interpolate(Vec3f(randUniform(),randUniform(),randUniform()));
        printf("%f %f %f\n",result.x,result.y,result.z);
    }

    return 0;
}
