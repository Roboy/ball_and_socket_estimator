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
#include "tinyxml.h"
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;
//#define SHOWORIENTATION_MEASURED
//#define SHOWORIENTATION_CARDSFLOW
//#define SHOW_EULER
//#define SHOW_SENSOR0_COMPONENTS
#define SHOWMAGNITUDE_SENSOR0
#define SHOWMAGNITUDE_SENSOR1
#define SHOWMAGNITUDE_SENSOR2
#define SHOWMAGNITUDE_SENSOR3
//#define SHOWSENSOR0
//#define SHOWSENSOR1
//#define SHOWSENSOR2
//#define SHOWSENSOR3
//#define GENERATESTL


boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

float sensor_location_scale = 5;
float sensor_scale = 2;

int main (int argc, char** argv)
{
    TiXmlDocument doc("/home/letrend/workspace/roboy3/src/robots/neck_magentic_field/cardsflow.xml");
    if (!doc.LoadFile()) {
        printf("Can't parse file\n");
        return false;
    }

    TiXmlElement *root = doc.RootElement();
    TiXmlElement *myoMuscle_it = NULL;
    vector<Vector3d> sensor_rel_locations;
    for (myoMuscle_it = root->FirstChildElement("myoMuscle"); myoMuscle_it;
         myoMuscle_it = myoMuscle_it->NextSiblingElement("myoMuscle")) {
        if (myoMuscle_it->Attribute("name")) {
            // myoMuscle joint acting on
            TiXmlElement *link_child_it = NULL;
            for (link_child_it = myoMuscle_it->FirstChildElement("link"); link_child_it;
                 link_child_it = link_child_it->NextSiblingElement("link")) {
                string link_name = link_child_it->Attribute("name");
                if (!link_name.empty()) {
                    TiXmlElement *viaPoint_child_it = NULL;
                    for (viaPoint_child_it = link_child_it->FirstChildElement("viaPoint"); viaPoint_child_it;
                         viaPoint_child_it = viaPoint_child_it->NextSiblingElement("viaPoint")) {
                        float x, y, z;
                        if (sscanf(viaPoint_child_it->GetText(), "%f %f %f", &x, &y, &z) != 3) {
                            printf("parser", "error reading [via point] (x y z)\n");
                            return false;
                        }
                        Vector3d local_coordinates(x, y, z);
                        sensor_rel_locations.push_back(local_coordinates);
                    }
                }
            }
        }
    }
    // Load input file into a PointCloud<T> with an appropriate type
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    FILE*       file = fopen("/home/letrend/workspace/roboy3/head_data0.log","r");

    if (NULL == file) {
        printf("Failed to open 'yourfile'");
        return -1;
    }
    fscanf(file, "%*[^\n]\n", NULL);
    float roll, pitch, yaw;
    float s[4][3];
    bool first = true;
//    Quaterniond quat_init;
    while(fscanf(file,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
    &s[0][0],&s[0][1],&s[0][2],
    &s[1][0],&s[1][1],&s[1][2],
    &s[2][0],&s[2][1],&s[2][2],
    &s[3][0],&s[3][1],&s[3][2],&roll,&pitch,&yaw)==15) {
        Vector3d dir(0, 0, 1);
        Vector3d mag0(s[0][0], s[0][1], s[0][2]);
        Vector3d mag1(s[1][0], s[1][1], s[1][2]);
        Vector3d mag2(s[2][0], s[2][1], s[2][2]);
        Vector3d mag3(s[3][0], s[3][1], s[3][2]);

//        Quaterniond q(qw, qx, qy, qz);

//        Matrix3d rot = q.matrix();
        Matrix3d rot;
        rot = AngleAxisd(roll, Vector3d::UnitX())
        * AngleAxisd(pitch, Vector3d::UnitY())
        * AngleAxisd(yaw, Vector3d::UnitZ());

#ifdef SHOWORIENTATION_MEASURED
        dir = rot * dir;
//            mag = rot*mag*0.01;
//            for(int sensor=0;sensor<3;sensor++){
        pcl::PointXYZRGB p0;
        p0.x = dir[0];
        p0.y = dir[1];
        p0.z = dir[2];
        p0.r = 255;
        p0.g = 255;
        p0.b = 255;
        cloud->push_back(p0);
#endif
#ifdef SHOWORIENTATION_CARDSFLOW
        dir << 0,0,1;
        Quaterniond q2(q_top_w, q_top_x, q_top_y, q_top_z);
//        if(first){
//            quat_init = q2;
//            first = false;
//        }
//        q2 = q2*quat_init.inverse();
        Matrix3d rot2 = q2.matrix();
        dir = rot2 * dir;
        p0.x = dir[0];
        p0.y = dir[1];
        p0.z = dir[2];
        p0.r = 0;
        p0.g = 30;
        p0.b = 90;
        cloud->push_back(p0);
#endif
#ifdef SHOW_EULER
        if(abs(roll)<0.5 && abs(pitch)<0.5 && abs(yaw)<1.5) {
            pcl::PointXYZRGB p;
            p.x = roll;
            p.y = pitch;
            p.z = yaw;
            p.r = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOW_SENSOR0_COMPONENTS
        {
            dir << 0, 0, 1 + mag0[0] * sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
        {
            dir << 0, 0, 1 + mag0[1] * sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
        {
            dir << 0, 0, 1 + mag0[2] * sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWMAGNITUDE_SENSOR0
        {
            double norm = mag0.norm();
            Vector3d dir = sensor_rel_locations[0]*norm*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWMAGNITUDE_SENSOR1
        {
            double norm = mag1.norm();
            Vector3d dir = sensor_rel_locations[1]*norm*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWMAGNITUDE_SENSOR2
        {
            double norm = mag2.norm();
            Vector3d dir = sensor_rel_locations[2]*norm*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWMAGNITUDE_SENSOR3
        {
            double norm = mag3.norm();
            Vector3d dir = sensor_rel_locations[3]*norm*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 235;
            p.g = 225;
            p.b = 52;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWSENSOR0
        {
            Vector3d dir = sensor_rel_locations[0]*sensor_location_scale + sensor_rel_locations[0]*mag0[0]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[0]*sensor_location_scale + sensor_rel_locations[0]*mag0[1]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[0]*sensor_location_scale + sensor_rel_locations[0]*mag0[2]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWSENSOR1
        {
            Vector3d dir = sensor_rel_locations[1]*sensor_location_scale + sensor_rel_locations[1]*mag1[0]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[1]*sensor_location_scale + sensor_rel_locations[1]*mag1[1]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[1]*sensor_location_scale + sensor_rel_locations[1]*mag1[2]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWSENSOR2
        {
            Vector3d dir = sensor_rel_locations[2]*sensor_location_scale + sensor_rel_locations[2]*mag2[0]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[2]*sensor_location_scale + sensor_rel_locations[2]*mag2[1]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[2]*sensor_location_scale + sensor_rel_locations[2]*mag2[2]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif
#ifdef SHOWSENSOR3
        {
            Vector3d dir = sensor_rel_locations[3]*sensor_location_scale + sensor_rel_locations[3]*mag3[0]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.r = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[3]*sensor_location_scale + sensor_rel_locations[3]*mag3[1]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.g = 255;
            cloud->push_back(p);
        }
        {
            Vector3d dir = sensor_rel_locations[3]*sensor_location_scale + sensor_rel_locations[3]*mag3[2]*sensor_scale;
            dir = rot * dir;
            pcl::PointXYZRGB p;
            p.x = dir[0];
            p.y = dir[1];
            p.z = dir[2];
            p.b = 255;
            cloud->push_back(p);
        }
#endif

    }


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis(cloud);

    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    //* the data should be available in cloud

#ifdef GENERATESTL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_xyz);

    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_xyz);
    n.setInputCloud (cloud_xyz);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    //* normals should not contain the point normals + surface curvatures

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud_xyz, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.5);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (10000);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    pcl::io::savePolygonFileSTL ("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/mesh.stl", triangles);
#endif
    // Finish
    return (0);
}