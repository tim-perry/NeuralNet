#ifndef NETWORK_H
#define NETWORK_H
#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include "dataset.hpp"

class Network {
//typedefs
	typedef Eigen::VectorXd (*activ_func_t)(Eigen::VectorXd winput);
	typedef double (*cost_func_t)(Eigen::VectorXd output, Eigen::VectorXd target);
	typedef Eigen::VectorXd (*cost_deriv_t)(Eigen::VectorXd output, Eigen::VectorXd target);


//attributes
public:
	std::vector<int> neurons;
	int layers;
	Eigen::MatrixXd* weight;
	Eigen::VectorXd* bias;
	
	std::vector<activ_func_t> activ_func;
    std::vector<activ_func_t> activ_deriv;
	cost_func_t cost_func;
	cost_deriv_t cost_deriv;
	
//constructors
public:
	Network(std::vector<int> vectorlayers);
	Network(const Network& network);
	~Network();

//methods
public:
	void save(const char*);
	static Network* load(const char*);
	void randomize(double factor = 1);
	Eigen::VectorXd feedForward(Eigen::VectorXd input);
	double cost(Dataset dataset);
	double accuracy(Dataset dataset);
	void train(Dataset dataset, int epochs, int batchsize, double lrate);

	Eigen::VectorXd clfeedforward(Eigen::VectorXd input);
	
private:
	Network* dNetwork(Eigen::VectorXd input, Eigen::VectorXd desired);
	void descend(Network* dnetwork);
	static void train_thread(std::mutex* mutex, Network* network, Network* netsum, Eigen::VectorXd input, Eigen::VectorXd desired);

    //activation functions 
    static Eigen::VectorXd sigmoid_a(Eigen::VectorXd z);
	static Eigen::VectorXd relu_a(Eigen::VectorXd z);
    static Eigen::VectorXd softmax_a(Eigen::VectorXd z);

    //activation function derivatives
    static Eigen::VectorXd sigmoid_ad(Eigen::VectorXd z);
    static Eigen::VectorXd relu_ad(Eigen::VectorXd z);
    static Eigen::VectorXd softmax_ad(Eigen::VectorXd z);

	//cost functions
	static double quadratic_c(Eigen::VectorXd output, Eigen::VectorXd target);
    static double cross_entropy_c(Eigen::VectorXd output, Eigen::VectorXd target);
    static double log_liklihood_c(Eigen::VectorXd ouput, Eigen::VectorXd target);

    //cost function derivatives
	static Eigen::VectorXd quadratic_cd(Eigen::VectorXd output, Eigen::VectorXd target);
    static Eigen::VectorXd cross_entropy_cd(Eigen::VectorXd output, Eigen::VectorXd target);
    static Eigen::VectorXd log_liklihood_cd(Eigen::VectorXd output, Eigen::VectorXd target);

    //Mathematical functions
    static double sigmoid(double x);
    static double sigmoid_d(double x);
    static double recip(double x);
    static double ln(double x);
    static double e(double x);
    static double neg(double x);
};

#endif