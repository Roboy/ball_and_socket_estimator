#pragma once

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "grid3d.hpp"

// std
#include <iostream>

using namespace Eigen;
using namespace std;
// Generic functor for Eigen Levenberg-Marquardt minimizer
template<typename _Scalar, int NX = Dynamic, int NY = Dynamic>
struct Functor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }

    int values() const { return m_values; }
};

struct PoseEstimator : Functor<double> {
    /**
     * Default amount of sensors needed for Eigen templated structure
     * @param numberOfSensors you can however choose any number of sensors here
     */
    PoseEstimator(int numberOfSensors = 4);

    /**
     * This is the function that is called in each iteration
     * @param x the pose vector (3 rotational parameters)
     * @param fvec the error function (the difference between the sensor positions)
     * @return
     */
    int operator()(const VectorXd &x, VectorXd &fvec) const;

    VectorXd pose;
    vector<Vector3d> sensor_pos, sensor_angle, sensor_target;
    int numberOfSensors = 4;
    Grid<float> *grid;
    float theta_range, theta_min;
};
