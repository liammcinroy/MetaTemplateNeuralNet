#pragma once

#include <map>
#include <math.h>
#include <time.h>

#include "matrix.h"
#include "layer.h"

class convolutional_neural_network
{
public:
	convolutional_neural_network(bool dropout);
	~convolutional_neural_network();
	float learning_rate;
	matrix<float> input;
	bool use_dropout;
	layer discriminate();//done
	layer generate(matrix<float> labels);//done
	void learn();//done
	void learn(matrix<float> labels);//done
	void push_layer(layer new_layer, bool maxpool = false, int rows = 0, int cols = 0);//done
private:
	std::vector<layer> m_layers;
	float max(float a, float b);//done
	float logistic(float x);//done
	matrix<float> logistic_regression(matrix<float> input_data);//done
	matrix<float> feed_forward(layer input_layer, unsigned int num_output_neurons);//done
	matrix<float> feed_backwards(layer input_layer, layer weights);//done
	matrix<float> convolve(matrix<float> input_matrix, matrix<float> kernel);//done
	matrix<float> convolve_backwards(layer input_layer, matrix<float> kernel, int feature_map);
	matrix<float> maxpool(matrix<float> input_matrix, unsigned int output_cols, unsigned int output_rows);//done
	float energy(layer visible, layer hidden);//done
	layer discriminate_to(unsigned int i);//done
	layer generate_to(unsigned int i, matrix<float> labels);//done
	void stochastic_gradient_descent(unsigned int i);//done
	void dropout(layer &input_layer);//done
	bool about_equals(matrix<float> first, matrix<float> second, float minimum_percent);//done
};

