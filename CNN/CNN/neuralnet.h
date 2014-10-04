#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <time.h>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

class NeuralNet
{
public:
	NeuralNet();
	~NeuralNet();
	//save learned net
	void save(std::string path);
	//load previously learned net
	void load(std::string path);
	//feed forwards
	ILayer* discriminate();
	//set input (batch will not be generated)
	void set_input(std::vector<Matrix<int>*> input);
	//set labels for batch
	void set_labels(Matrix<int>* batch_labels);
	//wake-sleep algorithm
	void pretrain();
	//backpropogate TODO!
	void train();
	//add a layer to the end of the network
	void add_layer(ILayer* layer);
	float learning_rate;
	bool use_dropout;
private:
	std::vector<ILayer*> layers;
	Matrix<int>* labels;
};