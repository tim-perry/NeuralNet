#include "network.hpp"
#include "dataset.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

using namespace Eigen;

//Basic Constructor
Network::Network(std::vector<int> neurons) {
	this->neurons = neurons;
	this->layers = neurons.size();

	weight = new MatrixXd[layers - 1];
	bias = new VectorXd[layers - 1];

	for (int i = 0; i < layers - 1; i++) {
		weight[i] = MatrixXd::Zero(neurons.at(i + 1), neurons.at(i));
		bias[i] = VectorXd::Zero(neurons.at(i + 1));

		activ_func.push_back(sigmoid_a);
		activ_deriv.push_back(sigmoid_ad);
	}

	cost_func = cross_entropy_c;
	cost_deriv = cross_entropy_cd;
}

//Constructor to create a copy of a network
Network::Network(const Network& network) : Network(network.neurons){
	for (int i = 0; i < layers - 1; i++) {
		this->weight[i] = network.weight[i];
		this->bias[i] = network.bias[i];
	}
}

//Destructor
Network::~Network() {
	delete[] weight;
	delete[] bias;
}

//Randomize all weights and biases of a network
void Network::randomize(double factor) {
	for (int i = 0; i < layers - 1; i++) {
		weight[i].setRandom() *= factor;
		bias[i].setRandom() *= factor;
	}
}

/*
void Network::save(const char* path) {
	std::ofstream* file = new std::ofstream(path, std::ios::binary);
	file->write((char*)&layers, sizeof(int));
	file->write((char*)neurons, layers * sizeof(int));
	
	for (int i = 0; i < layers - 1; i++) {
		int size = neurons[i + 1] * neurons[i] * sizeof(double);
		file->write((char*)weight[i].data(), size);
	}
	
	for (int i = 0; i < layers - 1; i++) {
		int size = neurons[i + 1] * sizeof(double);
		file->write((char*)bias[i].data(), size);
	}
	
	delete file;
}
*/
/*
Network* Network::load(const char* path) {
	std::ifstream* file = new std::ifstream(path, std::ios::binary);
	int layers;
	int* neurons = new int[layers];
    file->read((char*)&layers, sizeof(int));
	file->read((char*)neurons, layers * sizeof(int));

	Network* network = new Network(layers, neurons);

	for (int i = 0; i < layers - 1; i++) {
		int size = neurons[i + 1] * neurons[i] * sizeof(double);
		double* data = (double*)malloc(size);
		file->read((char*)data, size);
		memcpy(network->weight[i].data(), data, size);
		free(data);
	}

	for (int i = 0; i < layers - 1; i++) {
		int size = neurons[i + 1] * sizeof(double);
		double* data = (double*)malloc(size);
		file->read((char*)data, size);
		memcpy(network->bias[i].data(), data, size);
		free(data);
	}

	delete[] neurons;
	delete file;
	return network;
}
*/

//Feedforward an input into the neural network and return its output
VectorXd Network::feedForward(VectorXd activation) {
	for (int i = 0; i < layers - 1; i++) {
		activation = weight[i] * activation + bias[i];	//calculate weighted input
		activation = activ_func.at(i)(activation);		//apply activation function
	}
	return activation;	//return output vector
}

//Calculate average cost across a dataset
double Network::cost(Dataset dataset) {
	double cost = 0;
	for (int i = 0; i < dataset.size; i++) {
		VectorXd output = feedForward(dataset.examples.at(i));
		cost += cost_func(output, dataset.labels.at(i));
	}
	return cost / dataset.size;
}

//Calculate accuracy across a dataset (only applies to some problems)
double Network::accuracy(Dataset dataset) {
	int correct = 0;
	for (int i = 0; i < dataset.size; i++) {
		VectorXd output = feedForward(dataset.examples.at(i));

		//determine index of max value for both the output and target vectors
		int outmax = 0;
		int targmax = 0;
		for (int j = 0; j < output.rows(); j++) {
			if (output[j] > output[outmax]) outmax = j;
			if (dataset.labels.at(i)[j] > dataset.labels.at(i)[targmax]) targmax = j;
		}
		if (outmax == targmax) correct++; //correct if target matches output
	}
	return 100 * (double)correct / dataset.size;
}

//Gradient descent by delta network
void Network::descend(Network* dnetwork) {
	for (int i = 0; i < layers - 1; i++) {
		weight[i] -= dnetwork->weight[i];
		bias[i] -= dnetwork->bias[i];
	}
}

//Calculate derivatives across network and return a "delta network"
Network* Network::dNetwork(VectorXd input, VectorXd target) {
	Network* dnetwork = new Network(neurons);	//create delta network of same size
	VectorXd* activation = new VectorXd[layers];
	VectorXd* winput = new VectorXd[layers - 1];

	//move activation pointer so input is stored in index [-1] (hacky?)
	activation++;
	activation[-1] = input;

	//feedforward storing state and da/dz
	for (int i = 0; i < layers - 1; i++) {
		winput[i] = weight[i] * activation[i - 1] + bias[i];
		//activation[i] = winput[i].unaryExpr(func.activ);
		activation[i] = activ_func.at(i)(winput[i]);
		//winput[i] = winput[i].unaryExpr(func.activ_deriv); //Prepare DCDZ calculation by storing da/dz
		winput[i] = activ_deriv.at(i)(winput[i]);
	}
	
	//Backpropogate starting with the last layer
	for (int L = (layers - 2); 0 <= L; L--) {

		//CALCULATE DCDZ
		if (L == layers - 2) { //if last layer
			VectorXd dcda = cost_deriv(activation[L], target);
			winput[L].array() *= dcda.array(); //dcdz = dcda * dadz
		} else {
			//calculate derivative of z(L) with respect to activation(L - 1) (dz/da) equivalent step to finding biases of last
			MatrixXd dcda = winput[L + 1].asDiagonal() * weight[L + 1]; //calculate dc/dz * dz/da (dc/dz is previous winput and dz/da is weight in network)
			winput[L].array() *= dcda.colwise().sum().array(); //sum across neurons
		}
		
		//CALCULATE DCDB (dc/dz * dz/db)
		dnetwork->bias[L] = winput[L] * 1; //dz/db is always 1 as bias is a constant

		//CALCULATE DCDW (dc/dz * dz/dw)
		(dnetwork->weight[L]).colwise() = winput[L]; //load with dcdz to be multiplied with dzdb in next step
		dnetwork->weight[L] *= activation[L - 1].asDiagonal(); //equal to activation of neuron in previous layer
	}

	//memory cleanup
	activation--;
	delete[] activation;
	delete[] winput;

	return dnetwork;
}

//Train the neural network on a dataset
void Network::train(Dataset dataset, int epochs, int batchsize, double lrate) {
	if (batchsize == 0) batchsize = dataset.size; //batch gradient descent

	std::mutex* mutex = new std::mutex();
	//for each epoch
	for (int e = 0; e < epochs; e++) {
		//for each batch
		for (int i = 0; i < dataset.size; i += batchsize) {
			Network* netsum = new Network(neurons);
			std::thread worker[batchsize];
			
			for (int j = 0; j < batchsize; j++) {
				worker[j] = std::thread(train_thread, mutex, this, netsum, dataset.examples.at(i + j), dataset.labels.at(i + j));
			}

			for (int j = 0; j < batchsize; j++) {
				worker[j].join();
			}
		
			for (int j = 0; j < layers - 1; j++) {
				netsum->weight[j] *= lrate / batchsize;
				netsum->bias[j] *= lrate / batchsize;
			}
			descend(netsum);
			delete netsum;
		}
	}
	delete mutex;
}

void Network::train_thread(std::mutex* mutex, Network* network, Network* netsum, VectorXd input, VectorXd desired) {
	Network* dnetwork = network->dNetwork(input, desired);

	mutex->lock();
	for (int k = 0; k < network->layers - 1; k++) {
		netsum->weight[k] += dnetwork->weight[k];
		netsum->bias[k] += dnetwork->bias[k];
	}
	mutex->unlock();
	delete dnetwork;
}




//Activation functions
VectorXd Network::sigmoid_a(VectorXd z) {
	return z.unaryExpr(&sigmoid);
}

VectorXd Network::sigmoid_ad(VectorXd z) {
	return z.unaryExpr(&sigmoid_d);
}

VectorXd Network::softmax_a(VectorXd z) {
	double denom = z.unaryExpr(&e).sum();
	assert(denom != 0);
	for (int i = 0; i < z.rows(); i++) {
		z[i] = e(z[i]) / denom;
	}
	return z;
}

//softmax breaks everything because its derivative with respect to w relies on not just the weights in z'
VectorXd Network::softmax_ad(VectorXd z) {
	return VectorXd();
}

VectorXd Network::relu_a(VectorXd z) {
	for (int i = 0; i < z.rows(); i++) {
		if (z[i] < 0) z[i] = 0;
	}
	return z;
}

VectorXd Network::relu_ad(VectorXd z) {
	for (int i = 0; i < z.rows(); i++) {
		if (z[i] < 0) z[i] = 0;
		else z[i] = 1;
	}
	return z;
}


//Cost functions
double Network::quadratic_c(VectorXd output, VectorXd target) {
	VectorXd cost = output - target;
	cost = cost.array() * cost.array();
	return cost.sum();
}

VectorXd Network::quadratic_cd(VectorXd output, VectorXd target) {
    return 2 * (output - target);
}

double Network::cross_entropy_c(VectorXd output, VectorXd target) {
	VectorXd term1 = target.array() * output.unaryExpr(&ln).array();
	VectorXd term2 = target.unaryExpr(&recip).array() * output.unaryExpr(&recip).unaryExpr(&ln).array();
	VectorXd cost = term1 + term2;

	for (int i = 0; i < cost.rows(); i++) {
		if (std::isnan(cost[i])) cost[i] = 0;
	}

	return - cost.sum();
}

VectorXd Network::cross_entropy_cd(VectorXd output, VectorXd target) {
	VectorXd numerator = output - target;
	VectorXd denom = output.array() * output.unaryExpr(&recip).array();
	VectorXd result = numerator.array() / denom.array();

	for (int i = 0; i < result.rows(); i++)
		if (std::isnan(result[i])) result[i] = 0;

	return result;
}

double Network::log_liklihood_c(VectorXd output, VectorXd target) {
	VectorXd result = output.unaryExpr(&ln).array() * target.array();
	return - result.sum();
}

VectorXd Network::log_liklihood_cd(VectorXd output, VectorXd target) {
	VectorXd result = target.array() / output.array();
	return result.unaryExpr(&neg);
}


//Mathematical functions
//consider speed and accuracy in how this is implemented/optimized, along with any other derivative functions
double Network::sigmoid(double x) {
	return 1 / (1 + exp(-x));
}
double Network::sigmoid_d(double x) {
	double denom = 1 + exp(-x);
	return exp(-x) / (denom * denom);
}
double Network::recip(double x) {
	return 1 - x;
}
double Network::ln(double x) {
	return log(x);
}
double Network::e(double x) {
	return exp(x);
}
double Network::neg(double x) {
	return -x;
}