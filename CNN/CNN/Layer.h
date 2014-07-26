#pragma once

#include "matrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_OUTPUT 3

class layer
{
public:
	layer();
	layer(int num_feature_maps, int feature_rows, int feature_cols, int kind, int data_amount, int data_rows, int data_cols, int data_dims, bool use_random_values);
	layer(int num_feature_maps, int feature_rows, int feature_cols, int kind, int data_amount, int data_rows, int data_cols, int data_dims, float data_values);
	layer(int num_feature_maps, int feature_rows, int feature_cols, int kind, int data_amount, matrix<float> data_example, bool use_random_values);
	~layer();
	int feature_map_count;
	int data_count;
	int type;
	matrix<float> at(int i);
	void set_feature_maps(std::vector<matrix<float>> new_maps);
	float neuron_at(int f, int i, int j, int k);
	void set_neuron(int f, int i, int j, int k, float value);
	matrix<float> data_at(int i);
	matrix<float> data_at(int i, int k);
	void set_data(int i, matrix<float> value);
	void set_data(int i, std::vector<std::vector<std::vector<float>>> value);
	void set_data(int i, std::vector<std::vector<float>> value);
	float data_value_at(int f, int i, int j, int k);
	void set_data_value_at(int f, int i, int j, int k, float value);
	layer operator-(layer subtracted);
private:
	std::vector<matrix<float>> m_feature_maps;
	std::vector<matrix<float>> m_data;
};

