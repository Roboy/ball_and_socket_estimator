#pragma once

#include <ros/ros.h>
#include <stdio.h>
#include <map>
#include <roboy_middleware_msgs/MagneticSensor.h>
#include <common_utilities/CommonDefinitions.h>
#include <common_utilities/rviz_visualization.hpp>
#include <fstream>
#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>
#include "grid3d.hpp"
#include "cnpy.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <yaml-cpp/yaml.h>
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
#include "pose_estimator.hpp"

using namespace std;
using namespace Eigen;

float randUniform() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class BallInSocketEstimator : public rviz_visualization {

public:
    BallInSocketEstimator(string sensor_pos_filepath, string sensor_values_filepath, string config_filepath);

    ~BallInSocketEstimator();

    void magneticSensorCallback(const roboy_middleware_msgs::MagneticSensorConstPtr& msg);
private:
  bool fileExists(const string &filepath){
      struct stat buffer;
      return (stat (filepath.c_str(), &buffer) == 0);
  }

  template <typename T>
  vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
  }
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
private:
    ros::NodeHandlePtr nh;
    ros::Time start_time;
    boost::shared_ptr<ros::AsyncSpinner> spinner;
    ros::Subscriber magnetic_sensor_sub;
    boost::shared_ptr<PoseEstimator> estimator;
};
