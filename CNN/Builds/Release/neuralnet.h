#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

class NeuralNet
{
public:
	NeuralNet() = default;
	~NeuralNet();
	//save learned net
	void save_data(std::string path);
	//load previously learned net
	void load_data(std::string path);
	//feed forwards
	ILayer* discriminate();
	//set input (batch will not be generated)
	void set_input(std::vector<Matrix<float>*> input);
	//set labels for batch
	void set_labels(std::vector<Matrix<float>*> batch_labels);
	//wake-sleep algorithm
	void pretrain(int epochs);
	//backpropogate
	void train(int epochs);
	//add a layer to the end of the network
	void add_layer(ILayer* layer);
	float learning_rate;
	bool use_dropout;
	bool binary_net;
private:
	float global_error();
	float output_error_signal(int &i, int &j, int &k);
	float error_signal(int &i, int &j, int &k, float &weights_sum);
	Matrix2D<int, 4, 1>* coords(int &l, int &k, int &i, int &j);
	std::vector<ILayer*> layers;
	std::vector<Matrix<float>*> labels;
};