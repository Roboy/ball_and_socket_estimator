#include "ball_in_socket_estimator/ball_in_socket_estimator.hpp"

BallInSocketEstimator::BallInSocketEstimator(string sensor_pos_filepath, string sensor_values_filepath, string config_filepath) {
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketEstimator",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    nh.reset(new ros::NodeHandle);

    //load the entire npz file
    cnpy::npz_t sensor_positions_npz = cnpy::npz_load(sensor_pos_filepath);
    cnpy::npz_t sensor_values_npz = cnpy::npz_load(
            "/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/models/sensor_values.npz");

    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr_mv1 = sensor_positions_npz["values"];
    cnpy::NpyArray arr_mv2 = sensor_values_npz["values"];
    vector<vector<vector<double>>> sensor_position, sensor_values;

    int number_of_samples = arr_mv1.shape[0];
    int number_of_sensors = arr_mv1.shape[1]-1;

    convertFromNpy(arr_mv1.data<double>(), arr_mv1.shape, sensor_position);
    convertFromNpy(arr_mv2.data<double>(), arr_mv2.shape, sensor_values);

    {
      vector<vector<vector<double>>> sensor_position_new = sensor_position, sensor_values_new = sensor_values;
      // the sensors are interleaved
      vector<int> sort_order = {1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13};
      // sort
      for (int i = 0; i < number_of_samples; i++) {
          for (int j = 0; j < number_of_sensors; j++) {
              sensor_position_new[i][j] = sensor_position[i][sort_order[number_of_sensors-1-j]];
              sensor_values_new[i][j] = sensor_values[i][sort_order[number_of_sensors-1-j]];
          }
      }

      sensor_position = sensor_position_new;
      sensor_values = sensor_values_new;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    vector<vector<float>> phi, theta;
    phi.resize(number_of_sensors);
    theta.resize(number_of_sensors);
    for (int j = 0; j < number_of_sensors; j++) {
      for (int i = 0; i < number_of_samples; i++) {
          phi[j].push_back(atan2(sensor_position[i][j][2], sensor_position[i][j][0]));
          theta[j].push_back(atan2(sqrtf(powf(sensor_position[i][j][0], 2.0f) + powf(sensor_position[i][j][2], 2.0f)),
                                   sensor_position[i][j][1]));
          pcl::PointXYZRGB p;
          // p.x = sensor_position[i][j][0] / 100.0;
          // p.y = sensor_position[i][j][1] / 100.0;
          // p.z = sensor_position[i][j][2] / 100.0;
          // p.b = 255;
          // cloud->push_back(p);
          p.x = 0.22 * sin(theta[j][i]) * cos(phi[j][i]) + 0.0005 * sensor_values[i][j][0];
          p.y = 0.22 * cos(theta[j][i]) + 0.0005 * sensor_values[i][j][2];
          p.z = 0.22 * sin(theta[j][i]) * sin(phi[j][i]) + 0.0005 * sensor_values[i][j][1];
          p.b = 200;
          cloud->push_back(p);
      }
    }

    {
      // sort everything in increasing phi order
      vector<vector<float>> phi_new = phi, theta_new = theta;
      vector<vector<vector<double>>> sensor_position_new = sensor_position, sensor_values_new = sensor_values;
      for (int j = 0; j < number_of_sensors; j++) {
        auto indexes = sort_indexes<float>(phi[j]);
        for (int i = 0; i < number_of_samples; i++) {
          phi_new[j][i] = phi[j][indexes[i]];
          theta_new[j][i] = theta[j][indexes[i]];
          sensor_position_new[i][j] = sensor_position[indexes[i]][j];
          sensor_values_new[i][j] = sensor_values[indexes[i]][j];
          if(j==0){
            printf("%f\t",phi_new[j][i]);
          }
        }
      }
      phi = phi_new;
      theta = theta_new;
      sensor_position = sensor_position_new;
      sensor_values = sensor_values_new;
    }

    vector<float> theta_average;
    theta_average.resize(number_of_sensors);
    for (int i = 0; i < number_of_samples; i++) {
        for (int j = 0; j < number_of_sensors; j++) {
            theta_average[j] += theta[j][i];
        }
    }
    for (int j = 0; j < number_of_sensors; j++) {
        theta_average[j] /= number_of_samples;
    }

    printf("theta\n");
    for (int j = 0; j < number_of_sensors; j++) {
        printf("%f\n", theta_average[j]);
    }

    printf("theta step\n");
    for (int j = 0; j < number_of_sensors-1; j++) {
        printf("%f\n", theta_average[j+1]-theta_average[j]);
    }

    printf("phi\n");
    for (int j = 0; j < number_of_sensors; j++) {
        printf("%f\n", phi[j][0]);
    }

    float theta_step = (theta_average[1]-theta_average[0]);

    vector<vector<int>> phi_indices;
    vector<vector<float>> phi_min;
    phi_indices.resize(number_of_sensors);
    phi_min.resize(number_of_sensors);
    for (int j = 0; j < number_of_sensors; j++) {
        phi_indices[j].resize(360);
        phi_min[j].resize(360);
        for (int deg = 0; deg < 360; deg++) {
            phi_indices[j][deg] = 0;
            phi_min[j][deg] = 1000000;//arbirtary big
            for (int i = 0; i < number_of_samples; i++) {
                float dif = fabsf(phi[j][i]-((deg-180.0f) / 180.0f * M_PI));
                if (dif < phi_min[j][deg]) {
                    phi_min[j][deg] = dif;
                    phi_indices[j][deg] = i;
                }
            }
        }
    }

    for(int j=0;j<number_of_sensors;j++){
        for (int deg = 0; deg < 360; deg++) {
            pcl::PointXYZRGB p;
            p.x = 0.22 * sin(theta[j][0]) * cos(phi[j][phi_indices[j][deg]]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][0];
            p.y = 0.22 * cos(theta[j][0]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][2];
            p.z = 0.22 * sin(theta[j][0]) * sin(phi[j][phi_indices[j][deg]]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][1];
            p.g = 255;
            cloud->push_back(p);
            // if(j==1){
            //   printf("%f\t",phi_min[j][deg]);
            // }
        }
    }

    float theta_min = theta_average.front();
    float theta_range = (theta_average.back()-theta_average.front());

    if(!fileExists(config_filepath)) {
        ROS_FATAL_STREAM(config_filepath << " does not exist, check your path");
    }

    YAML::Node config;

    try{
      config = YAML::LoadFile(config_filepath);
    }catch(std::exception& e){
      ROS_ERROR_STREAM("yaml read exception in "<< config_filepath << " : " <<e.what());
    }

    vector<vector<float>> sensor_pos,sensor_angle;
    try{
      sensor_pos = config["sensor_pos"].as<vector<vector<float>>>();
      sensor_angle = config["sensor_angle"].as<vector<vector<float>>>();
    }catch(std::exception& e){
      ROS_ERROR_STREAM("yaml read exception in "<< config_filepath << " : " <<e.what());
    }

    estimator = boost::shared_ptr<PoseEstimator>(new PoseEstimator(sensor_select.size()));
    for(int i=0;i<sensor_select.size();i++){
      estimator->sensor_pos.push_back(Vector3d(sensor_pos[sensor_select[i]][0],sensor_pos[sensor_select[i]][1],sensor_pos[sensor_select[i]][2]));
      estimator->sensor_angle.push_back(Vector3d(sensor_angle[sensor_select[i]][0],sensor_angle[sensor_select[i]][1],sensor_angle[sensor_select[i]][2]));
      ROS_INFO_STREAM(estimator->sensor_pos.back().transpose());
    }

    estimator->theta_min = theta_min;
    estimator->theta_range = theta_range;

    estimator->grid = new Grid<float>(25,360,phi_indices,sensor_values);
    for (int i = 0; i < 360; i++) {
      for (int j = 0; j < number_of_sensors; j++) {
        // float phi_normalized = randUniform();
        // float theta_normalized = randUniform();
        float phi_normalized = (phi[j][phi_indices[j][i]]+M_PI)/(M_PI*2.0f);
        float theta_normalized = (theta[j][0]-theta_min)/theta_range;
        float theta_ = theta_normalized*theta_range+theta_min;
        float phi_ = phi_normalized*M_PI*2.0f-M_PI;
        // create a random location
        Vec3f result = estimator->grid->interpolate(theta_normalized,phi_normalized);
        // printf("%f %f %f\n",result.x,result.y,result.z);
        pcl::PointXYZRGB p;
        p.x = 0.22 * sin(theta_) * cos(phi_) + 0.0005 * result.x;
        p.y = 0.22 * cos(theta_) + 0.0005 * result.z;
        p.z = 0.22 * sin(theta_) * sin(phi_) + 0.0005 *result.y;
        p.r = 255;
        cloud->push_back(p);
      }
    }

    for (int i = 0; i < 100000; i++) {
        float phi_normalized = randUniform();
        float theta_normalized = randUniform();
        float theta_ = theta_normalized*theta_range+theta_min;
        float phi_ = phi_normalized*M_PI*2.0f-M_PI;
        // create a random location
        Vec3f result = estimator->grid->interpolate(theta_normalized,phi_normalized);
        // printf("%f %f %f\n",result.x,result.y,result.z);
        pcl::PointXYZRGB p;
        p.x = 0.22 * sin(theta_) * cos(phi_) + 0.0005 * result.x;
        p.y = 0.22 * cos(theta_) + 0.0005 * result.z;
        p.z = 0.22 * sin(theta_) * sin(phi_) + 0.0005 *result.y;
        p.r = 255;
        p.b = 255;
        cloud->push_back(p);
    }

    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis(cloud);
    //
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }



    spinner.reset(new ros::AsyncSpinner(0));
    spinner->start();

    magnetic_sensor_sub = nh->subscribe("/roboy/middleware/MagneticSensor", 100,
                                        &BallInSocketEstimator::magneticSensorCallback, this);
}

BallInSocketEstimator::~BallInSocketEstimator() {
}


void BallInSocketEstimator::magneticSensorCallback(const roboy_middleware_msgs::MagneticSensorConstPtr &msg) {
    ROS_INFO_THROTTLE(5, "receiving magnetic data");

    vector<Vector3d> sensor_target;
    for(int i=0;i<sensor_select.size();i++){
      Quaterniond quat(AngleAxisd(-estimator->sensor_angle[sensor_select[i]][2], Vector3d::UnitZ()));
      Vector3d mag(msg->x[sensor_select[i]],msg->y[sensor_select[i]],msg->z[sensor_select[i]]);
      mag = quat*mag;
      if(sensor_select[i]>=14)
        sensor_target.push_back(Vector3d(mag[0],-mag[1],-mag[2]));
      else
        sensor_target.push_back(mag);
    }
    estimator->sensor_target = sensor_target;

    NumericalDiff<PoseEstimator> *numDiff;
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PoseEstimator>, double> *lm;
    numDiff = new NumericalDiff<PoseEstimator>(*estimator);
    lm = new LevenbergMarquardt<NumericalDiff<PoseEstimator>, double>(*numDiff);
    lm->parameters.maxfev = 1000;
    lm->parameters.xtol = 1e-5;
    lm->parameters.ftol = 1e-5;
    VectorXd pose(3);
    pose << 0.1,0.1,0.1;
    int ret = lm->minimize(pose);
    ROS_INFO_THROTTLE(1,
                      "finished after %ld iterations, with an error of %f, result %.4f %.4f %.4f",
                      lm->iter,
                      lm->fnorm, pose[0], pose[1], pose[2]);
}

int main(int argc, char*argv[]) {

    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketEstimator",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    string sensor_pos_filepath("/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/models/sensor_position.npz");
    string sensor_values_filepath("/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/models/sensor_values.npz");
    string config_filepath("/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/configs/magnetic_field_calibration.yaml");

    BallInSocketEstimator ball_in_socket_estimator(sensor_pos_filepath,sensor_values_filepath,config_filepath);
    while(ros::ok()){
      ROS_INFO_THROTTLE(10,"running");
    }
    return 0;
}
