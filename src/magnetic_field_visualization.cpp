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

#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;


int main (int argc, char** argv)
{
    // Load input file into a PointCloud<T> with an appropriate type
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);


    FILE*       file = fopen("/home/letrend/workspace/roboy_control/data0.log","r");

    if (NULL == file) {
        printf("Failed to open 'yourfile'");
        return -1;
    }
    fscanf(file, "%*[^\n]\n", NULL);
    for(int i =0; i<210000; i++){
        float qx,qy,qz,qw, s[3][3], q_top_x,q_top_y, q_top_z, q_top_w;
        int nItemsRead = fscanf(file,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
        &qx,&qy,&qz,&qw,&s[0][0],&s[0][1],&s[0][2],&s[1][0],&s[1][1],&s[1][2],&s[2][0],&s[2][1],&s[2][2],&q_top_x,&q_top_y, &q_top_z, &q_top_w);
        if(nItemsRead ==17){
            Vector3d dir(0,0,1);
            Vector3d mag(s[2][0],s[2][1],s[2][2]);
            Quaterniond q(qw,qx,qy,qz);
            Matrix3d rot = q.matrix();
            dir = rot*dir;
            mag = rot*mag*0.01;
//            for(int sensor=0;sensor<3;sensor++){
            pcl::PointXYZ p0(dir[0],dir[1],dir[2]);
            cloud->push_back(p0);
            pcl::PointXYZ p(dir[0]+mag[0],dir[1]+mag[1],dir[2]+mag[2]);
            cloud->push_back(p);
//            }

        }
    }

    pcl::visualization::CloudViewer viewer("magnetic data");

    //blocks until the cloud is actually rendered
    viewer.showCloud(cloud);

    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer

    while (!viewer.wasStopped ())
    {
        //you can also do cool processing here
        //FIXME: Note that this is running in a separate thread from viewerPsycho
        //and you should guard against race conditions yourself...
    }
    //* the data should be available in cloud

    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    //* normals should not contain the point normals + surface curvatures

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    pcl::io::savePolygonFileSTL ("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/mesh.stl", triangles);

    // Finish
    return (0);
}