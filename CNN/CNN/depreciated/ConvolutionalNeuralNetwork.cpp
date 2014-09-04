#include "ConvolutionalNeuralNetwork.h"

convolutional_neural_network::convolutional_neural_network(bool dropout)
{
	m_layers = std::vector<layer>();
	use_dropout = dropout;
	srand(time(NULL));
}

convolutional_neural_network::~convolutional_neural_network()
{
}

void convolutional_neural_network::push_layer(layer new_layer, bool maxpool, int rows, int cols)
{
	new_layer.use_maxpool = maxpool;
	new_layer.maxpool_rows = rows;
	new_layer.maxpool_cols = cols;
	m_layers.push_back(new_layer);
}

layer convolutional_neural_network::discriminate()
{
	return discriminate_to(m_layers.size());
}

layer convolutional_neural_network::generate(matrix<float> labels)
{
	return generate_to(0, labels);
}

void convolutional_neural_network::learn(matrix<float> labels)
{
	while (true)
	{
		for (int i = 0; i < m_layers.size(); ++i)
			stochastic_gradient_descent(i);

		if (about_equals(discriminate().at(0), labels, .8f))
			break;
	}
}

matrix<float> convolutional_neural_network::convolve(matrix<float> input_matrix, matrix<float> kernel)
{
	int M = (kernel.cols - 1) / 2;
	int N = (kernel.rows - 1) / 2;
	matrix<float> result(input_matrix.cols - (2 * M), input_matrix.rows - (2 * N), 1);

	for (int k = 0; k < input_matrix.dims; ++k)
	{
		matrix<float> current(input_matrix.cols - (2 * M), input_matrix.rows - (2 * N), 1);
		//apply to every pixel
		for (int i = M; i < input_matrix.rows - N; ++i)
		{
			for (int j = N; j < input_matrix.cols - M; ++j)
			{
				//find sum
				float sum = 0.0f;
				for (int m = M; m >= -M; --m)
				for (int n = N; n >= -N; --n)
					sum += (input_matrix.at(i + m, j + n, k) * kernel.at(M + m, N + n, 0));
				current.set(i - M, j - N, 0, sum);
			}
		}

		//add channels
		for (int i = 0; i < result.rows; ++i)
		for (int j = 0; j < result.cols; ++j)
			result.set(i, j, 0, result.at(i, j, 0) + current.at(i, j, 0));
	}
	return result;
}

matrix<float> convolutional_neural_network::feed_forward(layer input_layer, unsigned int num_output)
{
	matrix<float> result(1, num_output, 1);

	for (int i = 0; i < num_output; ++i)
	{
		//Find sum of weights and run through ReL
		float sum = 0.0f;
		for (int j = 0; j < input_layer.at(0).rows; ++j)
			sum += (input_layer.neuron_at(0, j, 0, 0) * input_layer.data_value_at(i, j, 0, 0));
		result.set(i, 0, 0, logistic(sum));
	}
	return result;
}

matrix<float> convolutional_neural_network::feed_backwards(layer input_layer, layer weights)
{
	matrix<float> result(1, weights.at(0).cols, 1);

	for (int i = 0; i < weights.at(0).cols; ++i)
	{
		for (int j = 0; j < input_layer.at(0).rows; ++j)
			result.set(i, 0, 0, result.at(i, 0, 0) + (weights.data_value_at(j, i, 0, 0) * input_layer.neuron_at(0, j, 0, 0)));
		result.set(i, 0, 0, logistic(result.at(i, 0, 0)));
	}
	return result;
}

matrix<float> convolutional_neural_network::convolve_backwards(layer input_layer, matrix<float> kernel, int feature_map)
{
	matrix<float> result(input_layer.at(0).cols + (kernel.cols - 1), input_layer.at(0).rows + (kernel.rows - 1), 1);
	int M = (kernel.cols - 1) / 2;
	int N = (kernel.rows - 1) / 2;

	for (int i = N; i < input_layer.at(feature_map).rows - N; ++i)
	{
		for (int j = M; j < input_layer.at(feature_map).cols - M; ++j)
		{
			for (int i2 = i - N; i2 < i + N; ++i2)
			{
				for (int j2 = j - M; j2 < j + M; ++j2)
				{
					if (i2 - N < 0 || j2 - M < 0)
						continue;

					result.set(i, j, 0, result.at(i, j, 0) + (input_layer.neuron_at(feature_map, i - N, j - M, 0) 
						* kernel.at(i2 + N - i, j2 + M - j, 0)));
				}
			}
		}
	}
	return result;
}

layer convolutional_neural_network::discriminate_to(unsigned int i)
{
	layer current = m_layers[0];
	std::vector<matrix<float>> first_layer;
	for (int j = 0; j < input.dims; ++j)
		first_layer.push_back(input.at_channel(j));
	current.set_feature_maps(first_layer);

	for (int j = 0; j < i; ++j)
	{
		std::vector<matrix<float>> feature_maps;
		switch (current.type)
		{
		case CNN_CONVOLUTION:
			for (int i2 = 0; i2 < current.feature_map_count; ++i2)
			{
				matrix<float> total = convolve(current.at(i2), current.data_at(0));
				for (int k = 1; k < current.data_count; ++k)
					total += convolve(current.at(i2), current.data_at(k));
				feature_maps.push_back(total);
			}
			
			current = m_layers[j + 1];
			current.set_feature_maps(feature_maps);
			if (use_dropout)
				dropout(current);
			if (current.use_maxpool)
				current = current.maxpool();
			break;
		case CNN_FEED_FORWARD:
			feature_maps.push_back(feed_forward(current, m_layers[j + 1].at(0).cols));
			current = m_layers[j + 1];
			current.set_feature_maps(feature_maps);
			if (use_dropout)
				dropout(current);
			if (current.use_maxpool)
				current = current.maxpool();
			break;
		case CNN_OUTPUT:
			//current.set_feature_maps({ logistic_regression(current.at(0)) });
			return current;
			break;
		}
	}
	return current;
}

layer convolutional_neural_network::generate_to(unsigned int i, matrix<float> labels)
{
	layer current = m_layers[m_layers.size() - 1];
	current.set_feature_maps({ labels });
	for (int j = m_layers.size() - 1; j > i; --j)
	{
		try
		{
			std::vector<matrix<float>> feature_maps;
			switch (m_layers[j - 1].type)
			{
			case CNN_CONVOLUTION:
				for (int i2 = 0; i2 < current.feature_map_count; ++i2)
					feature_maps.push_back(convolve_backwards(current, m_layers[j - 1].data_at(i2), i2));

				current = m_layers[j - 1];
				current.set_feature_maps(feature_maps);
				if (use_dropout)
					dropout(current);
				break;
			case CNN_FEED_FORWARD:
				feature_maps.push_back(feed_backwards(current, m_layers[j - 1]));
				current = m_layers[j - 1];
				current.set_feature_maps(feature_maps);
				if (use_dropout)
					dropout(current);
				break;
			}
		}
		catch (int exception)
		{
			return current;
		}
	}
	return current;
}

void convolutional_neural_network::stochastic_gradient_descent(unsigned int i)
{
	layer discriminated = m_layers[i];
	layer generated;
	layer reconstructed;

	std::vector<matrix<float>> feature_maps;
	switch (m_layers[i].type)
	{
	case CNN_CONVOLUTION:
		//discriminate
		for (int i2 = 0; i2 < discriminated.feature_map_count; ++i2)
		{
			matrix<float> total = convolve(discriminated.at(i2), discriminated.data_at(0));
			for (int k = 1; k < discriminated.data_count; ++k)
				total += convolve(discriminated.at(i2), discriminated.data_at(k));
			feature_maps.push_back(total);
		}

		discriminated = m_layers[i + 1];
		discriminated.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(discriminated);
		if (discriminated.use_maxpool)
			discriminated = discriminated.maxpool();
		break;
	case CNN_FEED_FORWARD:
		//discriminate
		feature_maps.push_back(feed_forward(discriminated, m_layers[i + 1].at(0).cols));
		discriminated = m_layers[i + 1];
		discriminated.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(discriminated);
		if (discriminated.use_maxpool)
			discriminated = discriminated.maxpool();
		break;
	}

	feature_maps.clear();
	switch (m_layers[i + 1].type)
	{
	case CNN_CONVOLUTION:
		//generate
		for (int i2 = 0; i2 < discriminated.feature_map_count; ++i2)
			feature_maps.push_back(convolve_backwards(discriminated, m_layers[i].data_at(i2), i2));

		generated = m_layers[i];
		generated.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(generated);

		feature_maps.clear();

		//reconstruct
		for (int i2 = 0; i2 < generated.feature_map_count; ++i2)
		{
			matrix<float> total = convolve(generated.at(i2), generated.data_at(0));
			for (int k = 1; k < generated.data_count; ++k)
				total += convolve(generated.at(i2), generated.data_at(k));
			feature_maps.push_back(total);
		}

		reconstructed = m_layers[i + 1];
		reconstructed.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(reconstructed);
		if (reconstructed.use_maxpool)
			reconstructed = reconstructed.maxpool();
		break;
	case CNN_FEED_FORWARD:
		//generate
		feature_maps.push_back(feed_backwards(generated, m_layers[i + 1]));
		generated = m_layers[i];
		generated.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(generated);

		feature_maps.clear();

		//reconstruct
		feature_maps.push_back(feed_forward(generated, m_layers[i + 1].at(0).cols));
		reconstructed = m_layers[i + 1];
		reconstructed.set_feature_maps(feature_maps);
		if (use_dropout)
			dropout(reconstructed);
		if (reconstructed.use_maxpool)
			reconstructed = reconstructed.maxpool();
		break;
	}

	for (int i = 0; i < m_layers[i].data_count; ++i)
	{
		for (int j = 0; j < m_layers[i].at(0).rows; ++j)
		{
			m_layers[i].set_data_value_at(i, j, 0, 0, (m_layers[i].data_value_at(i, j, 0, 0) +
				(learning_rate * (discriminated.neuron_at(0, i, 0, 0) - reconstructed.neuron_at(0, i, 0, 0)))));
		}
	}
}

void convolutional_neural_network::dropout(layer& input_layer)
{
	for (int f = 0; f < input_layer.feature_map_count; ++f)
		for (int k = 0; k < input_layer.at(f).dims; ++k)
			for (int i2 = 0; i2 < input_layer.at(f).rows; ++i2)
				for (int j = 0; j < input_layer.at(f).cols; ++j)
					if (rand() % 10 < 5)
						input_layer.set_neuron(f, i2, j, k, 0.0f);
}

float convolutional_neural_network::max(float a, float b)
{
	return (a > b) ? a : b;
}

float convolutional_neural_network::energy(layer visible, layer hidden)
{
	float result = 0.0f;
	for (int f = 0; f < visible.feature_map_count; ++f)
		for (int k = 0; k < visible.at(f).dims; ++k)
			for (int i = 0; i < visible.at(f).rows; ++i)
				for (int j = 0; j < visible.at(f).cols; ++j)
					result += visible.neuron_at(f, i, j, k);
	for (int f = 0; f < hidden.feature_map_count; ++f)
		for (int k = 0; k < hidden.at(f).dims; ++k)
			for (int i = 0; i < hidden.at(f).rows; ++i)
				for (int j = 0; j < hidden.at(f).cols; ++j)
					result -= hidden.neuron_at(f, i, j, k);
	for (int i = 0; i < hidden.at(0).rows; ++i)
		for (int j = 0; j < visible.at(0).rows; ++j)
			result -= (visible.neuron_at(0, j, 0, 0) * visible.data_value_at(i, j, 0, 0) * hidden.neuron_at(0, i, 0, 0));

	return result;
}

matrix<float> convolutional_neural_network::logistic_regression(matrix<float> input_data)
{
	//add exp at some point
	matrix<float> result(1, input_data.rows, 1);

	float sum = 0.0f;
	for (int j = 0; j < input_data.rows; ++j)
		sum += input_data.at(j, 0, 0);

	for (int j = 0; j < input_data.rows; ++j)
		result.set(j, 0, 0, input_data.at(j, 0, 0) / sum);
	return result;
}

float convolutional_neural_network::logistic(float x)
{
	return 1 / (1 + exp(-x));
}

bool convolutional_neural_network::about_equals(matrix<float> first, matrix<float> second, float minimum)
{
	int count_correct = 0;
	int count_incorrect = 0;

	for (int k = 0; k < first.dims; ++k)
	{
		for (int i = 0; i < first.rows; ++i)
		{
			for (int j = 0; j < first.cols; ++j)
			{
				if (first.at(i, j, k) == second.at(i, j, k))
					++count_correct;
				else
					++count_incorrect;
			}
		}
	}

	return (count_correct / (count_correct + count_incorrect)) >= minimum;
}