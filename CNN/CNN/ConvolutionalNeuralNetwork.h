#pragma once

#include "matrix.h"
#include "layer.h"

class convolutional_neural_network
{
public:
	convolutional_neural_network();
	~convolutional_neural_network();
	float learning_rate;
	matrix<float> input;
	bool use_dropout;
	layer discriminate();
	layer generate(matrix<float> labels);
	void learn();
	void learn(matrix<float> labels);
private:
	std::vector<layer> m_layers;
	float max(float a, float b);//done
	matrix<float> logistic_regression(matrix<float> input_data);//done
	matrix<float> feed_forward(layer input_layer, int num_output_neurons);//done
	matrix<float> feed_backwards(layer input_layer, layer weights);//done
	matrix<float> convolve(matrix<float> input_matrix, matrix<float> kernal);//done
	matrix<float> deconvolve(matrix<float> input_matrix, matrix<float> kernal);//done
	matrix<float> deconvolve_single(float input_value, matrix<float> kernal);//done
	matrix<float> maxpool(matrix<float> input_matrix, int output_cols, int output_rows);//done
	layer discriminate_to(int i);//done
	layer generate_to(int i, matrix<float> labels);//done
	std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>> 
		stochastic_gradient_descent(int i, std::vector<matrix<float>> last_momentums);//done
	std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>
		stochastic_gradient_descent(int i, std::vector<matrix<float>> last_momentums, matrix<float> labels);//done
	layer dropout(layer input_layer);//done
};

