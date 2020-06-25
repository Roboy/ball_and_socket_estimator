#include "ball_in_socket_estimator/ball_in_socket_estimator.hpp"

BallInSocketEstimator::BallInSocketEstimator() {
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "BallInSocketEstimator",
                  ros::init_options::NoSigintHandler | ros::init_options::AnonymousName);
    }

    nh.reset(new ros::NodeHandle);
    spinner.reset(new ros::AsyncSpinner(0));

    magnetic_sensor_sub = nh->subscribe("/roboy/middleware/MagneticSensor", 100,
                                        &BallInSocketEstimator::magneticSensorCallback, this);
    visualization_pub = nh->advertise<visualization_msgs::Marker>("/visualization_marker", 1);
}

BallInSocketEstimator::~BallInSocketEstimator() {
}


void BallInSocketEstimator::magneticSensorCallback(const roboy_middleware_msgs::MagneticSensorConstPtr &msg) {
    ROS_DEBUG_THROTTLE(10, "receiving magnetic data");

}

void convertFromNpy(double *data, vector<unsigned long> shape, vector<vector<vector<double>>> &data_out) {
    data_out.resize(shape[0]);
    for (int i = 0; i < shape[0]; i++) {
        data_out[i].resize(shape[1]);
        for (int j = 0; j < shape[1]; j++) {
            data_out[i][j].resize(shape[2]);
            for (int k = 0; k < shape[2]; k++) {
                data_out[i][j][k] = data[i * shape[1] * shape[2] + j * shape[2] + k];
//              printf("%f\t",mv1[i * arr_mv1.shape[1] * arr_mv1.shape[2] + j * arr_mv1.shape[2] + k]);
            }
//          printf("\n");
        }
    }
}

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <stdio.h>
#include <pcl/visualization/cloud_viewer.h>

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

int main() {
    //load the entire npz file
    cnpy::npz_t sensor_positions_npz = cnpy::npz_load(
            "/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/models/sensor_position.npz");
    cnpy::npz_t sensor_values_npz = cnpy::npz_load(
            "/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/magjointlib/models/sensor_values.npz");

    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr_mv1 = sensor_positions_npz["values"];
    cnpy::NpyArray arr_mv2 = sensor_values_npz["values"];
    vector<vector<vector<double>>> sensor_position, sensor_values;

    int number_of_samples = arr_mv1.shape[0];
    int number_of_sensors = arr_mv1.shape[1];

    convertFromNpy(arr_mv1.data<double>(), arr_mv1.shape, sensor_position);
    convertFromNpy(arr_mv2.data<double>(), arr_mv2.shape, sensor_values);

    {
      vector<vector<vector<double>>> sensor_position_new = sensor_position, sensor_values_new = sensor_values;
      // the sensors are interleaved
      vector<int> sort_order = {0,1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13};
      // sort
      for (int i = 0; i < number_of_samples; i++) {
          for (int j = 0; j < number_of_sensors; j++) {
              sensor_position_new[i][j] = sensor_position[i][sort_order[j]];
              sensor_values_new[i][j] = sensor_values[i][sort_order[j]];
          }
      }
      sensor_position = sensor_position_new;
      sensor_values = sensor_values_new;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    vector<vector<float>> phi, theta;
    phi.resize(arr_mv1.shape[0]);
    theta.resize(arr_mv1.shape[0]);
    for (int i = 0; i < number_of_samples; i++) {
        for (int j = 0; j < number_of_sensors; j++) {
            phi[i].push_back(atan2(sensor_position[i][j][2], sensor_position[i][j][0]));
            theta[i].push_back(atan2(sqrtf(powf(sensor_position[i][j][0], 2.0f) + powf(sensor_position[i][j][2], 2.0f)),
                                     sensor_position[i][j][1]));
            pcl::PointXYZRGB p;
//            p.x = sensor_position[i][j][0] / 100.0;
//            p.y = sensor_position[i][j][1] / 100.0;
//            p.z = sensor_position[i][j][2] / 100.0;
//            p.r = 255;
//            cloud->push_back(p);
            p.x = 0.22 * sin(theta[i][j]) * cos(phi[i][j]) + 0.0005 * sensor_values[i][j][0];
            p.y = 0.22 * cos(theta[i][j]) + 0.0005 * sensor_values[i][j][2];
            p.z = 0.22 * sin(theta[i][j]) * sin(phi[i][j]) + 0.0005 * sensor_values[i][j][1];
            p.g = 255;
            cloud->push_back(p);
            // if(j==1){
            //   printf("%f\t",phi[i][j]);
            // }
        }
    }

    vector<float> theta_average;
    theta_average.resize(number_of_sensors);
    for (int i = 0; i < number_of_samples; i++) {
        for (int j = 0; j < number_of_sensors; j++) {
            theta_average[j] += theta[i][j];
        }
    }
    for (int j = 0; j < number_of_sensors; j++) {
        theta_average[j] /= number_of_samples;
    }

    printf("theta\n");
    for (int j = 0; j < number_of_sensors; j++) {
        printf("%f\n", theta_average[j]);
    }
    printf("phi\n");
    for (int j = 0; j < number_of_sensors; j++) {
        printf("%f\n", phi[0][j]);
    }

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
                float dif = fabsf(phi[i][j]-((deg-180.0f) / 180.0f * M_PI));
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
            p.x = 0.22 * sin(theta[0][j]) * cos(phi[phi_indices[j][deg]][j]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][0];
            p.y = 0.22 * cos(theta[0][j]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][2];
            p.z = 0.22 * sin(theta[0][j]) * sin(phi[phi_indices[j][deg]][j]) + 0.0005 * sensor_values[phi_indices[j][deg]][j][1];
            p.b = 255;
            cloud->push_back(p);
            // if(j==1){
            //   printf("%f\t",phi_min[j][deg]);
            // }
        }
    }

    float theta_range = (theta_average.front()-theta_average.back());

    Grid<float> grid(26,360,phi_indices,sensor_values);
    for (unsigned i = 0; i < 1000000; ++i) {
        float phi_normalized = randUniform();
        float theta_normalized = randUniform();
        // create a random location
        Vec3f result = grid.interpolate(theta_normalized,phi_normalized);
        // printf("%f %f %f\n",result.x,result.y,result.z);
        pcl::PointXYZRGB p;
        p.x = -0.22 * sin(theta_normalized*theta_range) * cos(phi_normalized*M_PI*2.0f) + 0.0005 * result.x;
        p.y = -0.22 * cos(theta_normalized*theta_range) + 0.0005 * result.y;
        p.z = -0.22 * sin(theta_normalized*theta_range) * sin(phi_normalized*M_PI*2.0f) + 0.0005 *result.z;
        p.r = 255;
        cloud->push_back(p);
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis(cloud);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}
