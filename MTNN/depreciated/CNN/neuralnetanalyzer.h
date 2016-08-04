#include <vector>

#pragma once

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

class NeuralNetAnalyzer
{
public:
	//approximates the weight gradient numerically
	static std::vector<FeatureMap> approximate_weight_gradient(NeuralNet &net);
	//approximates the bias gradient numerically
	static std::vector<FeatureMap> approximate_bias_gradient(NeuralNet &net);
	//approximates the weight hessian numerically
	static std::vector<FeatureMap> approximate_weight_hessian(NeuralNet &net);
	//approximates the bias hessian numerically
	static std::vector<FeatureMap> approximate_bias_hessian(NeuralNet &net);
	//find mean gradient error
	static std::pair<float, float> mean_gradient_error(NeuralNet &net, std::vector<FeatureMap> observed_weights,
		std::vector<FeatureMap> observed_biases);
	//find mean hessian error WARNING NOT NECESSARILY ACCURATE
	static std::pair<float, float> mean_hessian_error(NeuralNet &net);
	//find mean proportional gradient error
	static std::pair<float, float> proportional_gradient_error(NeuralNet &net, std::vector<FeatureMap> observed_weights,
		std::vector<FeatureMap> observed_biases);
	//find mean proportional hessian error WARNING NOT NECESSARILY ACCURATE
	static std::pair<float, float> proportional_hessian_error(NeuralNet &net);
	//update sample
	static void add_point(float value);
	//calculate the expected error
	static float mean_error();
	//save error data
	static void save_mean_error(std::string path);
	static int sample_size;
private:
	static std::vector<float> sample;
	static std::vector<float> errors;
};
