#include "Layer.h"

layer::layer()
{
}

layer::layer(unsigned int num, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int amount, unsigned int rows = 1, 
	unsigned int cols = 1, unsigned int dims = 1, bool use_random_values = false)
{
	for (int i = 0; i < num; ++i)
		m_feature_maps.push_back(matrix<float>(feature_cols, feature_rows, 1));
	type = kind;
	data_count = amount;
	switch (kind)
	{
	case CNN_CONVOLUTION:
		if (use_random_values)
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims, rand()));
		else
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims));
		break;
	case CNN_FEED_FORWARD:
		if (use_random_values)
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims, rand()));
		else
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims, rand()));
		break;
	case CNN_OUTPUT:
		break;
	default:
		break;
	}
}

layer::layer(unsigned int num, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int amount, unsigned int rows = 1,
	unsigned int cols = 1, unsigned int dims = 1, float values = 0.0f)
{
	for (int i = 0; i < num; ++i)
		m_feature_maps.push_back(matrix<float>(feature_cols, feature_rows, 1));
	type = kind;
	data_count = amount;
	switch (kind)
	{
	case CNN_CONVOLUTION:
		for (int i = 0; i < amount; ++i)
			m_data.push_back(matrix<float>(cols, rows, dims, values));
		break;
	case CNN_FEED_FORWARD:
		for (int i = 0; i < amount; ++i)
			m_data.push_back(matrix<float>(cols, rows, dims, values));
		break;
	case CNN_OUTPUT:
		break;
	default:
		break;
	}
}

layer::layer(unsigned int num, unsigned int feature_rows, unsigned int feature_cols, unsigned int kind, unsigned int amount,
	matrix<float> example, bool use_random_values = false)
{
	for (int i = 0; i < num; ++i)
		m_feature_maps.push_back(matrix<float>(feature_cols, feature_rows, 1));
	type = kind;
	data_count = amount;

	int cols = example.cols;
	int rows = example.rows;
	int dims = example.dims;
	switch (kind)
	{
	case CNN_CONVOLUTION:
		if (use_random_values)
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims, rand()));
		else
			for (int i = 0; i < amount; ++i)
				m_data.push_back(example);
		break;
	case CNN_FEED_FORWARD:
		if (use_random_values)
			for (int i = 0; i < amount; ++i)
				m_data.push_back(matrix<float>(cols, rows, dims, rand()));
		else
			for (int i = 0; i < amount; ++i)
				m_data.push_back(example);
		break;
	case CNN_OUTPUT:
		break;
	default:
		break;
	}
}

layer::~layer()
{
}

matrix<float> layer::at(unsigned int i)
{
	return m_feature_maps[i];
}

void layer::set_feature_maps(std::vector<matrix<float>> new_maps)
{
	m_feature_maps = new_maps;
}

float layer::neuron_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k)
{
	return m_feature_maps[f].at(i, j, k);
}

void layer::set_neuron(unsigned int f, unsigned int i, unsigned int j, unsigned int k, float value)
{
	m_feature_maps[f].set(i, j, k, value);
}

matrix<float> layer::data_at(unsigned int i)
{
	return m_data[i];
}

matrix<float> layer::data_at(unsigned int i, unsigned int k)
{
	return m_data[i].at_channel(k);
}

void layer::set_data(std::vector<matrix<float>> value)
{
	m_data = value;
}

void layer::set_data(unsigned int i, matrix<float> value)
{
	m_data[i] = value;
}

void layer::set_data(unsigned int i, std::vector<std::vector<std::vector<float>>> value)
{
	m_data[i] = value;
}

void layer::set_data(unsigned int i, std::vector<std::vector<float>> value)
{
	m_data[i] = value;
}

float layer::data_value_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k)
{
	return m_data[f].at(i, j, k);
}

void layer::set_data_value_at(unsigned int f, unsigned int i, unsigned int j, unsigned int k, float value)
{
	m_data[f].set(i, j, k, value);
}

layer layer::operator-(layer subtacted)
{
	for (int f = 0; f < this->feature_map_count; ++f)
		for (int k = 0; k < this->at(f).dims; ++k)
			for (int i = 0; i < this->at(f).rows; ++i)
				for (int j = 0; j < this->at(f).cols; ++j)
					this->set_neuron(f, i, j, k, this->neuron_at(f, i, j, k) - subtacted.neuron_at(f, i, j, k));
	return (*this);
}