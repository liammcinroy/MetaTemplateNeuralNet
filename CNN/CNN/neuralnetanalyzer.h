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
	//find mean gradient error
	static std::pair<float, float> mean_gradient_error(NeuralNet &net, std::vector<std::vector<IMatrix<float>*>> observed_weights,
													std::vector<std::vector<IMatrix<float>*>> observed_biases);
	//update sample
	static void add_point(float value);
	//calculate the mean squared error
	static float mean_squared_error();
	//save mse data
	static void save_mean_square_error(std::string path);
	static int sample_size;
private:
	static std::vector<float> sample;
	static std::vector<float> mses;
};