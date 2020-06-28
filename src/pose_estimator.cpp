#include "ball_in_socket_estimator/pose_estimator.hpp"

PoseEstimator::PoseEstimator(int numberOfSensors) :
        Functor<double>(3, 3 * numberOfSensors), numberOfSensors(numberOfSensors) {
    pose = VectorXd(3);

}

int PoseEstimator::operator()(const VectorXd &x, VectorXd &fvec) const {
    // construct quaternion (cf unit-sphere projection Terzakis paper)
    double alpha_squared = pow(pow(x(0), 2.0) + pow(x(1), 2.0) + pow(x(2), 2.0), 2.0);
    Quaterniond q((1 - alpha_squared) / (alpha_squared + 1),
                  2.0 * x(0) / (alpha_squared + 1),
                  2.0 * x(1) / (alpha_squared + 1),
                  2.0 * x(2) / (alpha_squared + 1));
    q.normalize();
    vector<Vector3d> sensor_pos_new;
    vector<float> phi,theta,phi_,theta_;
    vector<Vec3<float>> sensor_value;
    phi.resize(sensor_pos.size());
    theta.resize(sensor_pos.size());
    phi_.resize(sensor_pos.size());
    theta_.resize(sensor_pos.size());
    sensor_value.resize(sensor_pos.size());
    float error = 0;
    for(int i=0;i<sensor_pos.size();i++){
      sensor_pos_new.push_back(q*sensor_pos[i]);
      phi[i] = atan2(sensor_pos_new[i][2], sensor_pos_new[i][0]);
      theta[i] = atan2(sqrtf(powf(sensor_pos_new[i][0], 2.0f) + powf(sensor_pos_new[i][2], 2.0f)),
                               sensor_pos_new[i][1]);
      phi_[i] = (phi[i]+M_PI)/(M_PI*2.0f);
      theta_[i] = (theta[i]-theta_min)/theta_range;
      sensor_value[i] = grid->interpolate(theta_[i],phi_[i]);
      error+= sqrtf(powf(sensor_target[i][0]-sensor_value[i][0],2.0f)+
                    powf(sensor_target[i][1]-sensor_value[i][1],2.0f)+
                    powf(sensor_target[i][2]-sensor_value[i][2],2.0f));
    }
    fvec[0] = error;
    fvec[1] = error;
    fvec[2] = error;

    return 0;
}
