#pragma once

#include "matrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_OUTPUT 3

class layer
{
public:
	layer();
	layer(unsigned int num_feature_maps, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int data_amount,
		unsigned int data_rows, unsigned int data_cols, unsigned int data_dims, bool use_random_values);
	layer(unsigned int num_feature_maps, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int data_amount,
		unsigned int data_rows, unsigned int data_cols, unsigned int data_dims, float data_values);
	layer(unsigned int num_feature_maps, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int data_amount,
		matrix<float> data_example, bool use_random_values);
	~layer();
	unsigned int feature_map_count;
	unsigned int data_count;
	unsigned int type;
	matrix<float> at(unsigned int i);
	void set_feature_maps(std::vector<matrix<float>> new_maps);
	float neuron_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k);
	void set_neuron(unsigned int f, unsigned int i, unsigned int j, unsigned int k, float value);
	matrix<float> data_at(unsigned int i);
	matrix<float> data_at(unsigned int i, unsigned int k);
	void set_data(std::vector<matrix<float>> value);
	void set_data(unsigned int i, matrix<float> value);
	void set_data(unsigned int i, std::vector<std::vector<std::vector<float>>> value);
	void set_data(unsigned int i, std::vector<std::vector<float>> value);
	float data_value_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k);
	void set_data_value_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k, float value);
	layer operator-(layer subtracted);
private:
	std::vector<matrix<float>> m_feature_maps;
	std::vector<matrix<float>> m_data;
};

