#include "dataset.hpp"
#include <fstream>
#include <string>
#include <Eigen/Dense>

//Default Constructor
Dataset::Dataset() {
    this->size = 0;
    this->examples = std::vector<Eigen::VectorXd>();
    this->labels = std::vector<Eigen::VectorXd>();
}

//only works if data file neurons have values given by a uint8_t
void Dataset::add_examples_bin(const char* path, int offset, int stride, int size) {
    //determine number of examples in file
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streampos filesize = file.tellg();
    int count = ((uint32_t)filesize - offset + 1) / stride;
    this->size += count;

    //loop read them into VectorXd and add to std::vector
    file.seekg(offset, std::ios::beg);
    uint8_t* buffer = (uint8_t*)malloc(size);
    for (int i = 0; i < count; i++) {
        file.read((char*)buffer, size);

        Eigen::VectorXd example(size);
        for (int n = 0; n < size; n++) {
            example[n] = (double)buffer[n] / 255;
        }

        this->examples.push_back(example);
        file.seekg(stride - size, std::ios::cur);
    }

    free(buffer);
    file.close();
}

void Dataset::add_labels_bin(const char* path, int offset, int stride, int size) {
    //determine number of labels in file
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streampos filesize = file.tellg();
    int count = ((uint32_t)filesize - offset + 1) / stride;

    //loop read them into VectorXd and add to std::vector
    file.seekg(offset, std::ios::beg);
    Eigen::VectorXd label(size);
    uint8_t index;
    for (int i = 0; i < count; i++) {
        file.read((char*)&index, sizeof(uint8_t));
        label.setZero();
        label[index] = 1;
        this->labels.push_back(label);
        file.seekg(stride - sizeof(uint8_t), std::ios::cur);
    }

    file.close();
}








//Add content of directory to dataset and label
void Dataset::add_dir(const char* dir_path, Eigen::VectorXd label) {
    //+= size by number of things in directory
    //extract example vectors from image format and put in input examples array
    //add "label"s to labels array

}