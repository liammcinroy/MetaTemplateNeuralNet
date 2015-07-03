#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

#define CNN_QUADRATIC 0
#define CNN_CROSS_ENTROPY 1
#define CNN_LOG_LIKELIHOOD 2

class NeuralNet
{
public:
	NeuralNet() = default;
	~NeuralNet();
	//setup gradient
	void setup_gradient();
	//save learned net
	void save_data(std::string path);
	//load previously learned net
	void load_data(std::string path);
	//feed forwards
	void discriminate();
	//set input (batch will not be generated)
	void set_input(std::vector<IMatrix<float>*> &input);
	//set labels for batch
	void set_labels(std::vector<IMatrix<float>*> &batch_labels);
	//wake-sleep algorithm
	void pretrain(int iterations);
	//backpropogate with levenbourg-marquardt
	float train(int iterations);
	//backprop with custom gradients
	float train(int iterations, std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases);
	//add a layer to the end of the network
	void add_layer(ILayer* layer);
	//reset and apply gradient
	void apply_gradient();
	//apply custom gradient
	void apply_gradient(std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases);
	//get current error
	float global_error();
	float learning_rate;
	float momentum_term;
	int cost_function = CNN_QUADRATIC;
	bool use_dropout = false;
	bool use_batch_learning = false;
	bool use_momentum = true;
	std::vector<ILayer*> layers;
	std::vector<IMatrix<float>*> input;
	std::vector<IMatrix<float>*> labels;
	std::vector<std::vector<IMatrix<float>*>> weight_gradient;
	std::vector<std::vector<IMatrix<float>*>> bias_gradient;
private:
	std::vector<std::vector<IMatrix<float>*>> weight_momentum;
	std::vector<std::vector<IMatrix<float>*>> bias_momentum;
	Matrix2D<int, 4, 1>* coords(int &l, int &k, int &i, int &j);
	void dropout(ILayer* &layer);
	//TODO: FIX
	int error_signals();
};