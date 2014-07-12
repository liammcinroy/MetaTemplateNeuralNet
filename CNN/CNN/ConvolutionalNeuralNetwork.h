#pragma once

#include <map>
#include <string>

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
	void discriminate();
	void generate(matrix<float> input_matrix);
	void learn();
	matrix<float> augment(matrix<float> input_matrix);
private:
	std::vector<layer> m_layers;
	float max(float a, float b);//done
	matrix<float> logistic_regression(matrix<float> input_data);//done
	matrix<float> feed_forward(layer input_layer, int num_output_neurons);//done
	matrix<float> convolve(matrix<float> input_matrix, matrix<float> kernal);//done
	matrix<float> deconvolve(matrix<float> input_matrix, matrix<float> kernal);//done
	matrix<float> maxpool(matrix<float> input_matrix, int output_cols, int output_rows);//done
	layer discriminate_to(int i);//done
	layer generate_until(int i, matrix<float> input_matrix);
	void backpropogate(int i);
	float backpropogate_difference(int i);
	void stochastic_gradient_descent(int i);
	void dropout(int i);
};

