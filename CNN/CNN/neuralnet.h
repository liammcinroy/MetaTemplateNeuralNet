#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

class NeuralNet
{
public:
	//save learned net
	virtual void save(const std::string &path);
	//load previously learned net
	virtual void load(const std::string &path);
	//feed forwards
	ILayer discriminate();
	//set input (batch will not be generated)
	void set_input(Matrix2D<int>* input);
	//set labels for batch
	void set_labels(Matrix2D<int> batch_labels);
	//wake-sleep algorithm
	void pretrain();
	//backpropogate TODO!
	void train();
	float learning_rate;
	bool use_dropout;
	std::vector<ILayer> layers;
	Matrix2D<int> labels;
};