#ifndef DATASET_H
#define DATASET_H

#include <Eigen/Dense>
#include <vector>

class Dataset {
public:
    //attributes
    int size;
    std::vector<Eigen::VectorXd> examples;
    std::vector<Eigen::VectorXd> labels;

    //constructors
    Dataset();

    //methods
    void add_examples_bin(const char* path, int offset, int stride, int size);
    void add_labels_bin(const char* path, int offset, int stride, int size);
    void add_dir(const char* dir_path, Eigen::VectorXd label);
    
    //methods for synthesizing new data?
};

#endif