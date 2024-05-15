#include "compute.hpp"
#include "network.hpp"
#include <iostream>
#include <Eigen/Dense>

Compute* Compute::instance = NULL;

Compute* Compute::get() {
    if (instance == NULL) {
        instance = new Compute(0, 0);
    }
    return instance;
}

void Compute::list() {
    cl::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
    
    for (int i = 0; i < (int)platformList.size(); i++) {
        std::cout << "Platform " << i << std::endl;
        std::string info;
        platformList.at(i).getInfo(CL_PLATFORM_NAME, &info);
        std::cout << "Name: " << info << std::endl;
        platformList.at(i).getInfo(CL_PLATFORM_VENDOR, &info);
        std::cout << "Vendor: " << info << std::endl;
        
        cl::vector<cl::Device> deviceList;
        platformList.at(i).getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
        
        std::cout << "Devices on platform:" << std::endl;
        for (int j = 0; j < (int)deviceList.size(); j++) {
            deviceList.at(j).getInfo(CL_DEVICE_NAME, &info);
            std::cout << info << std::endl;
        }
        std::cout << std::endl;
    }
}

Compute::Compute(int platform, int device) {
    instance = this;

    //Get device ID
    cl::vector<cl::Platform> platformList;
    cl::vector<cl::Device> deviceList;
	cl::Platform::get(&platformList);
    platformList.at(platform).getDevices(CL_DEVICE_TYPE_GPU, &deviceList);
    this->device_id = deviceList.at(device).get();

    //Create a context on the device
	this->context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
    if (!context) {
        std::cerr << "Error: Failed to create OpenCL context!" << std::endl;
        exit(-1);
    }

    //Create a command queue on the device
    properties[0] = CL_QUEUE_PROPERTIES;
    properties[1] = CL_QUEUE_PROFILING_ENABLE;
    properties[2] = 0;
    this->queue = clCreateCommandQueueWithProperties(context, device_id, properties, nullptr);
    if (!queue) {
        std::cerr << "Error: Failed to create OpenCL command queue!" << std::endl;
        exit(-1);
    }

    //Create kernel function
    std::string source = "\
        __kernel void matrix_multiply(__global double* input, int input_size, __global double* output, int output_size, __global double* matrix, __global double* bias) {\
            int row = get_global_id(0);\
            if (row < output_size) {\
                for (int i = 0; i < input_size; i++) {\
                    output[row] += input[i] * matrix[row + i * output_size];\
                }\
                output[row] += bias[row];\
            }\
        }\
    ";
    const char* source_ptr = source.c_str();
    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, nullptr, &err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    std::cout << "clBuildProgram(): " << err << std::endl;
    size_t build_log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
    char* build_log = new char[build_log_size];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, nullptr);
    clFinish(queue);
    std::cout << build_log << std::endl;

    matrix_mult = clCreateKernel(program, "matrix_multiply", &err);
    std::cout << "Kernel error: " << err << std::endl;
}

Compute::~Compute() {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

Eigen::VectorXd Network::clfeedforward(Eigen::VectorXd input) {
    //Get openCL context
    cl_context context = Compute::get()->context;
    cl_command_queue queue = Compute::get()->queue;
    cl_kernel matrix_mult = Compute::get()->matrix_mult;

    //Initialize activation vectors
    Eigen::VectorXd* activation = new Eigen::VectorXd[layers];
    activation[0] = input;
    for (int i = 1; i < layers; i++) {
        activation[i] = Eigen::VectorXd::Zero(neurons.at(i));
    }

    //Write activation vectors and network matrices into gpu memory
    cl_mem* vec_mem = new cl_mem[layers];
    cl_mem* mat_mem = new cl_mem[layers - 1];
    cl_mem* bias_mem = new cl_mem[layers - 1];

    for (int i = 0; i < layers; i++) {
        vec_mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, activation[i].rows() * sizeof(double), nullptr, nullptr);
        clEnqueueWriteBuffer(queue, vec_mem[i], CL_TRUE, 0, activation[i].rows() * sizeof(double), activation[i].data(), 0, nullptr, nullptr);
    }

    for (int i = 0; i < layers - 1; i++) {
        mat_mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, weight[i].rows() * weight[i].cols() * sizeof(double), nullptr, nullptr);
        clEnqueueWriteBuffer(queue, mat_mem[i], CL_TRUE, 0, weight[i].rows() * weight[i].cols() * sizeof(double), weight[i].data(), 0, nullptr, nullptr);

        bias_mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, bias[i].rows() * sizeof(double), nullptr, nullptr);
        clEnqueueWriteBuffer(queue, bias_mem[i], CL_TRUE, 0, bias[i].rows() * sizeof(double), bias[i].data(), 0, nullptr, nullptr);
    }

    //Perform feedforward
    for (int i = 0; i < layers - 1; i++) {
        clSetKernelArg(matrix_mult, 0, sizeof(vec_mem[i]), &vec_mem[i]);
        clSetKernelArg(matrix_mult, 1, sizeof(int), &(neurons.at(i)));
        clSetKernelArg(matrix_mult, 2, sizeof(vec_mem[i + 1]), &vec_mem[i + 1]);
        clSetKernelArg(matrix_mult, 3, sizeof(int), &(neurons.at(i + 1)));
        clSetKernelArg(matrix_mult, 4, sizeof(mat_mem[i]), &mat_mem[i]);
        clSetKernelArg(matrix_mult, 5, sizeof(bias_mem[i]), &bias_mem[i]);

        size_t global_size[1] = {(size_t)(neurons.at(i + 1))};
        clEnqueueNDRangeKernel(queue, matrix_mult, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
        //clFinish(queue); //should I?
    }

    //Read result and return
    Eigen::VectorXd output(neurons.at(layers - 1));
    clEnqueueReadBuffer(queue, vec_mem[layers - 1], CL_TRUE, 0, neurons.at(layers - 1) * sizeof(double), output.data(), 0, nullptr, nullptr);
    clFinish(queue);
    
    return output;
}
