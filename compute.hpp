#ifndef COMPUTE_H
#define COMPUTE_H

#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/opencl.hpp>
#include <Eigen/Dense>
#include "network.hpp"

class Compute {
private:
    static Compute* instance;
    cl_device_id device_id;
    cl_queue_properties properties[3];
    Compute(int platform_id, int device_id);
    ~Compute();

public:
    
    cl_context context;
    cl_command_queue queue;
    cl_kernel matrix_mult;

public:
    static Compute* get();
    static void list();
};

#endif