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

void convolutional_neural_network::learn()
{
	matrix<float> labels = discriminate().at(0);

	std::map<int, std::vector<matrix<float>>> momentums;

	while (true)
	{
		for (int i = 0; i < m_layers.size(); ++i)
		{
			std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>> stochastic_descent;
			if (momentums.find(i) != momentums.end())
			{
				stochastic_descent = stochastic_gradient_descent(i, momentums[i]);
				momentums[i] = stochastic_descent.second;
			}
			else
			{
				stochastic_descent = stochastic_gradient_descent(i);
				momentums.insert(std::pair<int, std::vector<matrix<float>>>(i, stochastic_descent.second));
			}

			m_layers[i].set_data(stochastic_descent.first);
		}

		if (about_equals(discriminate().at(0), labels, .8f))
			break;
	}
}

void convolutional_neural_network::learn(matrix<float> labels)
{
	std::map<int, std::vector<matrix<float>>> momentums;

	while (true)
	{
		for (int i = 0; i < m_layers.size(); ++i)
		{
			std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>> stochastic_descent;
			if (momentums.find(i) != momentums.end())
			{
				stochastic_descent = stochastic_gradient_descent(i, momentums[i]);
				momentums[i] = stochastic_descent.second;
			}
			else
			{
				stochastic_descent = stochastic_gradient_descent(i);
				momentums.insert(std::pair<int, std::vector<matrix<float>>>(i, stochastic_descent.second));
			}

			m_layers[i].set_data(stochastic_descent.first);
		}

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
				for (int m = -M; m <= M; ++m)
				for (int n = -N; n <= N; ++n)
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

matrix<float> convolutional_neural_network::deconvolve(matrix<float> input_matrix, matrix<float> kernel)
{
	int N = (kernel.cols - 1) / 2;
	int M = (kernel.rows - 1) / 2;
	matrix<float> result(input_matrix.cols + (2 * N), input_matrix.rows + (2 * N), 1, INFINITY);

	matrix<float> top_left = deconvolve_single(input_matrix.at(0, 0, 0), kernel);
	for (int i = 0; i < kernel.rows; ++i)
	for (int j = 0; j< kernel.cols; ++j)
	if (top_left.at(i, j, 0) != 0)
		result.set(i, j, 0, top_left.at(i, j, 0));

	for (int i = 0; i < input_matrix.rows; ++i)
	{
		for (int j = 1; j < input_matrix.cols; ++j)
		{
			matrix<float> current_matrix = deconvolve_single(input_matrix.at(i, j, 0), kernel);
			matrix<float> new_kernel = kernel;

			float overlap_sum = 0.0f;
			int new_i = i + N;
			int new_j = j + M;
			int n = 0;
			int m = 0;

			for (int i2 = new_i - N; i2 <= new_i + N; ++i2)
			{
				for (int j2 = new_j - M; j2 <= new_j + M; ++j2)
				{
					if (result.at(i2, j2, 0) != INFINITY)
					{
						overlap_sum += kernel.at(n, m, 0) * result.at(i2, j2, 0);
						new_kernel.set(n, m, 0, 0);
					}
					++m;
				}
				m = 0;
				++n;
			}

			matrix<float> new_matrix = deconvolve_single(input_matrix.at(i, j, 0) - overlap_sum, new_kernel);

			n = 0;
			m = 0;
			for (int i2 = new_i - N; i2 <= new_i + N; ++i2)
			{
				for (int j2 = new_j - M; j2 <= new_j + M; ++j2)
				{
					if (new_matrix.at(n, m, 0) != 0)
						result.set(i2, j2, 0, new_matrix.at(n, m, 0));
					++m;
				}
				m = 0;
				++n;
			}
		}
	}

	for (int i = 0; i < result.rows; ++i)
	for (int j = 0; j < result.cols; ++j)
	if (result.at(i, j, 0) == INFINITY)
		result.set(i, j, 0, 0);

	return result;
}

matrix<float> convolutional_neural_network::deconvolve_single(float input_value, matrix<float> kernel)
{
	int N = (kernel.cols - 1) / 2;
	int M = (kernel.rows - 1) / 2;
	int symmetry = 0;

	matrix<float> result(kernel.cols, kernel.rows, 1);
	std::vector<float> p;
	std::vector<float> s;

	std::vector<std::pair<int, int>> matched;

	for (int i = 0; i < kernel.rows; ++i)
	{
		for (int j = 0; j < kernel.cols; ++j)
		{
			std::pair<int, int> coords(i, j);

			std::pair<int, int> matched_coords;
			if (kernel.at(i, j, 0) == -kernel.at(kernel.cols - 1 - j, kernel.rows - 1 - i, 0))//bottom left to right diagonal
			{
				symmetry = 1;
				matched_coords = std::pair<int, int>(kernel.cols - i - j, kernel.rows - 1 - i);
			}
			else if (kernel.at(i, j, 0) == -kernel.at(j, kernel.rows - 1 - i, 0))//bottom right to left diagonal
			{
				symmetry = 2;
				matched_coords = std::pair<int, int>(j, kernel.rows - 1 - i);
			}
			else if (kernel.at(i, j, 0) == -kernel.at(i, kernel.rows - 1 - j, 0))//across
			{
				symmetry = 3;
				matched_coords = std::pair<int, int>(i, kernel.rows - 1 - j);
			}
			else if (kernel.at(i, j, 0) == -kernel.at(kernel.cols - 1 - i, j, 0))//up and down
			{
				symmetry = 4;
				matched_coords = std::pair<int, int>(kernel.cols - 1 - i, j);
			}

			bool matched_before = false;
			for (int l = 0; l < matched.size(); ++l)
			{
				if (matched[l] == matched_coords || matched[l] == coords)
				{
					matched_before = true;
					break;
				}
			}

			if (symmetry != 0 && kernel.at(i, j, 0) != 0 && !matched_before &&
				((kernel.at(i, j, 0) > 0 && input_value > 0) || (kernel.at(i, j, 0) < 0 && input_value < 0)))
			{
				p.push_back(kernel.at(i, j, 0));
				matched.push_back(matched_coords);
				matched.push_back(coords);
			}

			if ((kernel.at(i, j, 0) > 0 && input_value > 0) || (kernel.at(i, j, 0) < 0 && input_value < 0))
				s.push_back(kernel.at(i, j, 0));
			symmetry = 0;
		}
	}

	float sum_P = 0.0f;
	float sum_S = 0.0f;

	for (int i = 0; i < p.size(); ++i)
		sum_P += abs(p[i]);
	for (int i = 0; i < s.size(); ++i)
		sum_S += abs(s[i]);

	//p
	float P = sum_P / sum_S;

	//k
	std::vector<float> k;
	for (int i = 0; i < kernel.rows; ++i)
	for (int j = 0; j < kernel.cols; ++j)
		k.push_back(kernel.at(i, j, 0));
	//m_n
	std::vector<float> m(k.size());
	for (int n = 0; n < k.size(); ++n)
	{
		bool in_p = false;
		bool in_s = false;

		for (int i = 0; i < p.size(); ++i)
		{
			if (p[i] == k[n])
			{
				in_p = true;
				break;
			}
		}

		for (int i = 0; i < s.size(); ++i)
		{
			if (s[i] == k[n])
			{
				in_s = true;
				break;
			}
		}

		//algorithms
		if (k[n] == 0)
			m[n] = 0;
		else if (in_p && in_s)
			m[n] = abs(((P * input_value) / p.size()) / k[n]);
		else if (!in_p && in_s)
			m[n] = abs(((1 - P) * input_value) / k[n]);
		else if (!in_s)
			m[n] = abs(1 / k[n]);
	}

	float sum = 0.0f;
	for (int i = 0; i < k.size(); ++i)
		sum += k[i] * m[i];

	float C = 0.0f;
	if (sum != 0)
		C = input_value / sum;

	int n = 0;
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			result.set(i, j, 0, abs(m[n] * C));
			++n;
		}
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
		result.set(i, 0, 0, max(0, sum));
	}
	return result;
}

matrix<float> convolutional_neural_network::feed_backwards(layer input_layer, layer weights)
{
	matrix<float> result(1, weights.at(0).cols, 1);

	for (int i = 0; i < weights.at(0).cols; ++i)
	{
		//Find sum of weights and run through ReL
		float sum = 0.0f;
		for (int j = 0; j < weights.data_count; ++j)
			sum += (input_layer.neuron_at(0, j, 0, 0) * weights.data_value_at(j, i, 1, 0));
		result.set(i, 0, 0, max(0, sum));
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
				current = dropout(current);
			if (current.use_maxpool)
				current = current.maxpool();
			break;
		case CNN_FEED_FORWARD:
			feature_maps.push_back(feed_forward(current, m_layers[j + 1].at(0).cols));
			current = m_layers[j + 1];
			current.set_feature_maps(feature_maps);
			if (use_dropout)
				current = dropout(current);
			if (current.use_maxpool)
				current = current.maxpool();
			break;
		case CNN_OUTPUT:
			current.set_feature_maps({ logistic_regression(current.at(0)) });
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
				for (int i2 = 0; i2 = current.feature_map_count; ++i2)
				{
					matrix<float> total = deconvolve(current.at(i2), m_layers[j].data_at(0));
					for (int k = 1; k < current.data_count; ++k)
						total += deconvolve(current.at(i2), m_layers[j].data_at(k));
					feature_maps.push_back(total);
				}
				current = m_layers[j - 1];
				current.set_feature_maps(feature_maps);
				if (use_dropout)
					current = dropout(current);
				//TODO: Handle max pooling
				break;
			case CNN_FEED_FORWARD:
				feature_maps.push_back(feed_backwards(current, m_layers[j - 1]));
				current = m_layers[j - 1];
				current.set_feature_maps(feature_maps);
				if (use_dropout)
					current = dropout(current);
				//TODO: Handle max pooling
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

std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>
convolutional_neural_network::stochastic_gradient_descent(unsigned int i)
{
	std::vector<matrix<float>> last_momentums;

	matrix<float> labels = discriminate().at(0);
	layer discriminated = discriminate_to(i);
	layer generated = generate_to(i, labels);
	layer subtracted = discriminated - generated;

	float sum = 0.0f;
	int neuron_count = subtracted.feature_map_count * subtracted.at(0).dims * subtracted.at(0).rows * subtracted.at(0).cols;

	for (int f = 0; f < subtracted.feature_map_count; ++f)
	for (int k = 0; k < subtracted.at(f).dims; ++k)
	for (int i = 0; i < subtracted.at(f).rows; ++i)
	for (int j = 0; j < subtracted.at(f).cols; ++j)
		sum += subtracted.neuron_at(f, i, j, k);

	float average_neuron_offset = sum / neuron_count;

	std::vector<matrix<float>> new_weights;

	for (int d = 0; d < subtracted.data_count; ++d)
	{
		new_weights.push_back(matrix<float>(subtracted.data_at(d).cols, subtracted.data_at(d).rows, subtracted.data_at(d).dims));
		last_momentums.push_back(matrix<float>(subtracted.data_at(d).cols, subtracted.data_at(d).rows, subtracted.data_at(d).dims));
		for (int k = 0; k < subtracted.data_at(d).dims; ++k)
		{
			for (int i = 0; i < subtracted.data_at(d).rows; ++i)
			{
				for (int j = 0; j < subtracted.data_at(d).cols; ++j)
				{
					float new_momentum = (0.9f * 1) - (0.0005f * learning_rate * subtracted.data_value_at(d, i, j, k))
						- (learning_rate * average_neuron_offset);
					float new_weight = subtracted.data_value_at(d, i, j, k) + new_momentum;
					last_momentums[d].set(i, j, k, new_momentum);
					new_weights[d].set(i, j, k, new_weight);
				}
			}
		}
	}
	return std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>(new_weights, last_momentums);
}

std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>> 
convolutional_neural_network::stochastic_gradient_descent(unsigned int i, std::vector<matrix<float>> last_momentums)
{
	matrix<float> labels = discriminate().at(0);
	layer discriminated = discriminate_to(i);
	layer generated = generate_to(i, labels);
	layer subtracted = discriminated - generated;
	
	float sum = 0.0f;
	int neuron_count = subtracted.feature_map_count * subtracted.at(0).dims * subtracted.at(0).rows * subtracted.at(0).cols;

	for (int f = 0; f < subtracted.feature_map_count; ++f)
		for (int k = 0; k < subtracted.at(f).dims; ++k)
			for (int i = 0; i < subtracted.at(f).rows; ++i)
				for (int j = 0; j < subtracted.at(f).cols; ++j)
					sum += subtracted.neuron_at(f, i, j, k);

	float average_neuron_offset = sum / neuron_count;

	std::vector<matrix<float>> new_weights;

	for (int d = 0; d < subtracted.data_count; ++d)
	{
		new_weights.push_back(matrix<float>(subtracted.data_at(d).cols, subtracted.data_at(d).rows, subtracted.data_at(d).dims));

		for (int k = 0; k < subtracted.data_at(d).dims; ++k)
		{
			for (int i = 0; i < subtracted.data_at(d).rows; ++i)
			{
				for (int j = 0; j < subtracted.data_at(d).cols; ++j)
				{
					float new_momentum = (0.9f * last_momentums[d].at(i, j, k)) - (0.0005f * learning_rate * subtracted.data_value_at(d, i, j, k))
						- (learning_rate * average_neuron_offset);
					float new_weight = subtracted.data_value_at(d, i, j, k) + new_momentum;
					last_momentums[d].set(i, j, k, new_momentum);
					new_weights[d].set(i, j, k, new_weight);
				}
			}
		}
	}
	return std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>(new_weights, last_momentums);
}

std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>
convolutional_neural_network::stochastic_gradient_descent(unsigned int i, std::vector<matrix<float>> last_momentums, matrix<float> labels)
{
	layer discriminated = discriminate_to(i);
	layer generated = generate_to(i, labels);
	layer subtracted = discriminated - generated;

	float sum = 0.0f;
	int neuron_count = subtracted.feature_map_count * subtracted.at(0).dims * subtracted.at(0).rows * subtracted.at(0).cols;

	for (int f = 0; f < subtracted.feature_map_count; ++f)
	for (int k = 0; k < subtracted.at(f).dims; ++k)
	for (int i = 0; i < subtracted.at(f).rows; ++i)
	for (int j = 0; j < subtracted.at(f).cols; ++j)
		sum += subtracted.neuron_at(f, i, j, k);

	float average_neuron_offset = sum / neuron_count;

	std::vector<matrix<float>> new_weights;

	for (int d = 0; d < subtracted.data_count; ++d)
	{
		new_weights.push_back(matrix<float>(subtracted.data_at(d).cols, subtracted.data_at(d).rows, subtracted.data_at(d).dims));

		for (int k = 0; k < subtracted.data_at(d).dims; ++k)
		{
			for (int i = 0; i < subtracted.data_at(d).rows; ++i)
			{
				for (int j = 0; j < subtracted.data_at(d).cols; ++j)
				{
					float new_momentum = (0.9f * last_momentums[d].at(i, j, k)) - (0.0005f * learning_rate * subtracted.data_value_at(d, i, j, k))
						- (learning_rate * average_neuron_offset);
					float new_weight = subtracted.data_value_at(d, i, j, k) + new_momentum;
					last_momentums[d].set(i, j, k, new_momentum);
					new_weights[d].set(i, j, k, new_weight);
				}
			}
		}
	}
	return std::pair<std::vector<matrix<float>>, std::vector<matrix<float>>>(new_weights, last_momentums);
}

layer convolutional_neural_network::dropout(layer input_layer)
{
	layer result = input_layer;
	for (int f = 0; f < input_layer.feature_map_count; ++f)
		for (int k = 0; k < input_layer.at(f).dims; ++k)
			for (int i2 = 0; i2 < input_layer.at(f).rows; ++i2)
				for (int j = 0; j < input_layer.at(f).cols; ++j)
					if (rand() % 10 < 5)
						result.set_neuron(f, i2, j, k, 0.0f);
	return result;
}

float convolutional_neural_network::max(float a, float b)
{
	return (a > b) ? a : b;
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

bool convolutional_neural_network::about_equals(matrix<float> first, matrix<float> second, float minimum_percent)
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

	return (count_correct / (count_correct + count_incorrect)) >= minimum_percent;
}