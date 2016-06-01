#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

#define CNN_COST_QUADRATIC 0
#define CNN_COST_CROSSENTROPY 1
#define CNN_COST_LOGLIKELIHOOD 2
#define CNN_COST_TARGETS 3

#define CNN_OPT_BACKPROP 0
#define CNN_OPT_ADAM 1
#define CNN_OPT_ADAGRAD 2

class NeuralNet
{
public:
	NeuralNet() = default;
	~NeuralNet();
	//create an exact deep copy of network
	NeuralNet copy_network();
	//setup gradient
	void setup_gradient();
	//save learned net
	void save_data(std::string path);
	//load previously learned net
	void load_data(std::string path);
	//feed forwards
	void discriminate();
	//feed backwards
	FeatureMap generate();
	//set input (batch will not be generated)
	void set_input(const FeatureMap &input);
	//set labels for batch
	void set_labels(const FeatureMap &batch_labels);
	//wake-sleep algorithm
	void pretrain(int iterations);
	//backpropogate with levenbourg-marquardt
	float train(int iterations, float mse);
	//backprop with custom gradients
	float train(int iterations, std::vector<FeatureMap> weights, std::vector<FeatureMap> biases, float mse);
	//update second derivatives
	void calculate_hessian(bool use_first_deriv, float gamma);
	//add a layer to the end of the network
	void add_layer(ILayer* layer);
	//reset and apply gradient
	void apply_gradient();
	//apply custom gradient
	void apply_gradient(std::vector<FeatureMap> weights, std::vector<FeatureMap> biases);
	//get current error
	float global_error();
	
	//Parameters
	
	float learning_rate = .1f;
	float minimum_divisor = .1f;
	float momentum_term = .8f;
	float beta1 = .9f;
	float beta2 = .99f;
	float epsilon = .0000001f;
	int cost_function = CNN_COST_QUADRATIC;
	int optimization_method = CNN_OPT_BACKPROP;
	bool use_dropout = false;
	bool use_batch_learning = false;
	bool use_momentum = false;
	bool use_hessian = false;
	
	int t = 0;
	std::vector<ILayer*> layers;
	FeatureMap input;
	FeatureMap labels;
	std::vector<FeatureMap> weight_gradient;
	std::vector<FeatureMap> biases_gradient;
private:
	std::vector<FeatureMap> weight_momentum;
	std::vector<FeatureMap> biases_momentum;
	Matrix2D<int, 4, 1>* coords(int &l, int &k, int &i, int &j);
	void dropout(ILayer* &layer);
	//TODO: FIX
	int error_signals();
	int hessian_error_signals();
};