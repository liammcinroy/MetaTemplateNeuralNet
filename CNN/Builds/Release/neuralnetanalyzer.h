#include <vector>

#pragma once

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

class NeuralNetAnalyzer
{
public:
	//approximates the weight gradient numerically
	static std::vector<std::vector<IMatrix<float>*>> approximate_weight_gradient(NeuralNet &net);
	//approximates the bias gradient numerically
	static std::vector<std::vector<IMatrix<float>*>> approximate_bias_gradient(NeuralNet &net);
	//update sample
	static void add_point(float value);
	//calculate the mean squared error
	static float mean_squared_error();
	static int sample_size;
private:
	static std::vector<float> sample;
};

