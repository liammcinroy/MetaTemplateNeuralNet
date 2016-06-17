#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

#define CNN_LOSS_QUADRATIC 0
#define CNN_LOSS_CROSSENTROPY 1
#define CNN_LOSS_LOGLIKELIHOOD 2
#define CNN_LOSS_TARGETS 3

#define CNN_OPT_BACKPROP 0
#define CNN_OPT_ADAM 1
#define CNN_OPT_ADAGRAD 2

class NeuralNet
{
public:
	NeuralNet() = default;
	~NeuralNet();
	//create an exact deep copy of network; still must call setup_gradient() on copy network
	NeuralNet copy_network();
	//setup gradient, must do
	void setup_gradient();
	//save learned net
	void save_data(std::string path);
	//load previously learned net
	void load_data(std::string path);
	//feed forwards
	void discriminate();
	//feed backwards, returns a copy of the first layer (must be deallocated)
	FeatureMap generate();
	//set input (batch will not be generated)
	void set_input(const FeatureMap &input);
	//set labels for batch
	void set_labels(const FeatureMap &batch_labels);
	//wake-sleep algorithm
	void pretrain(int iterations);
	//backpropogate with selected method, returns error by loss function
	float train();
	//backprop with custom gradients, returns error by loss function
	float train(std::vector<FeatureMap> weights, std::vector<FeatureMap> biases);
	//update second derivatives
	void calculate_hessian(bool use_first_deriv, float gamma);
	//add a layer to the end of the network
	void add_layer(ILayer* layer);
	//reset and apply gradient
	void apply_gradient();
	//apply custom gradient
	void apply_gradient(std::vector<FeatureMap> weights, std::vector<FeatureMap> biases);
	//get current error according to loss function
	float global_error();
	
	//Hyperparameters
	
	//learning rate (should be positive)
	float learning_rate = .001f;
	//only set if using hessian
	float minimum_divisor = .1f;
	//only set if using momentum 
	float momentum_term = .8f;
	//only set if using dropout. This proportion of neurons will be "dropped"
	float dropout_probability = .5f;
	//must be set if using Adam
	float beta1 = .9f;
	//must be set if using Adam
	float beta2 = .99f;
	//must be set if using Adam
	float epsilon = .0000001f;
	//must be set
	int loss_function = CNN_LOSS_QUADRATIC;
	//must be set; Adam and Adagrad set use_momentum and use_hessian
	int optimization_method = CNN_OPT_BACKPROP;
	//must be set
	bool use_dropout = false;
	//must be set
	bool use_batch_learning = false;
	//cannot be true if using Adam or Adagrad
	bool use_momentum = false;
	//cannot be true if using Adam or Adagrad
	bool use_hessian = false;
	//must add a sample with calculate_batch_statistics() before training with the minibatch. Resets statistics after apply_gradient() call
	bool use_batch_normalization = false;
	//if enabled with batch normalization, then every call to discriminate() will be sampled and the statistics will never be reset
	bool keep_running_activation_statistics = false;
	//if enabled with batch normalization, then every call to train() and discrimate() will be sampled
	bool collect_data_while_training = false;

	std::vector<ILayer*> layers;
	FeatureMap input;
	FeatureMap labels;
	std::vector<FeatureMap> weights_gradient;
	std::vector<FeatureMap> biases_gradient;
private:
	//used for adam
	int t = 0;
	//used for batch normalization
	int n = 0;
	//used for keeping running statistics
	bool currently_training = false;

	//momentum data and used for adam
	std::vector<FeatureMap> weights_momentum;
	std::vector<FeatureMap> biases_momentum;

	//batch normalization data
	std::vector<FeatureMap> activations_mean;
	std::vector<FeatureMap> activations_variance;

	void dropout(ILayer* &layer);
	//TODO: FIX
	int error_signals();
	int hessian_error_signals();
};