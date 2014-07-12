#include "ConvolutionalNeuralNetwork.h"

convolutional_neural_network::convolutional_neural_network()
{
}

convolutional_neural_network::~convolutional_neural_network()
{
}

matrix<float> convolutional_neural_network::convolve(matrix<float> input_matrix, matrix<float> kernal)
{
	int M = (kernal.cols - 1) / 2;
	int N = (kernal.rows - 1) / 2;
	matrix<float> result = matrix<float>(input_matrix.cols - (2 * M), input_matrix.rows - (2 * N), 1);

	for (int k = 0; k < input_matrix.dims; ++k)
	{
		matrix<float> current = matrix<float>(input_matrix.cols - (2 * kernal.cols), input_matrix.rows - (2 * kernal.cols), 1);
		//apply to every pixel
		for (int i = M; i < input_matrix.cols - M; ++i)
		{
			for (int j = N; j < input_matrix.rows - N; ++j)
			{
				//find sum
				float sum = 0.0f;
				for (int m = -M; m <= M; ++m)
					for (int n = -N; n <= N; ++n)
						sum += (input_matrix.at(i + m, j + n, k) * kernal.at(M - m, N - n, 0));
				current.set(i - M, j - N, 1, sum);
			}
		}

		//add channels
		for (int i = 0; i < result.cols; ++i)
			for (int j = 0; j < result.rows; ++j)
				result.set(i, j, 0, result.at(i, j, 0) + current.at(i, j, 0));
	}
	return result;
}

matrix<float> convolutional_neural_network::deconvolve(matrix<float> input_matrix, matrix<float> kernal)
{
	int N = (kernal.cols - 1) / 2;
	int M = (kernal.rows - 1) / 2;
	matrix<std::string> map(kernal.cols, kernal.rows, 1);
	matrix<std::string> codes(input_matrix.cols + (2 * N), input_matrix.rows + (2 * M), 1);

	int symmetry = 0;
	if (kernal.at(1, 0, 0) == -kernal.at(kernal.cols - 1, kernal.rows - 2, 0))//bottom left to right diagonal
		symmetry = 1;
	else if (kernal.at(1, 0, 0) == -kernal.at(0, kernal.rows - 2, 0))//bottom right to left diagonal
		symmetry = 2;
	else if (kernal.at(1, 0, 0) == -kernal.at(1, kernal.rows - 1, 0))//across
		symmetry = 3;
	else if (kernal.at(1, 0, 0) == -kernal.at(kernal.cols - 2, 0, 0))//up and down
		symmetry = 4;

	bool pos_top = (kernal.at(0, 0, 0) != 0) ? (kernal.at(0, 0, 0) > 0) : ((kernal.at(1, 0, 0) != 0) ? (kernal.at(1, 0, 0) > 0) :
		((kernal.at(0, 1, 0) != 0) ? (kernal.at(0, 1, 0) > 0) : ((kernal.at(1, 1, 0) != 0) ? (kernal.at(1, 1, 0) > 0) : false)));
	int times_symmetrical = 1;
	int unknown_constants = 1;

	if (symmetry != 4 && symmetry != 0)
	{
		for (int i = 0; i < kernal.cols; ++i)
		{
			for (int j = 0; j < kernal.rows; ++j)
			{
				if (kernal.at(i, j, 0) == 0)
				{
					map.set(i, j, 0, "U" + unknown_constants);
					++unknown_constants;
					break;
				}

				else
				{
					switch (symmetry)
					{
					case 1:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + times_symmetrical);
							map.set(j, kernal.rows - 1 - i, 0, "S" + (-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + (-times_symmetrical));
							map.set(j, kernal.rows - 1 - i, 0, "S" + times_symmetrical);
							++times_symmetrical;
						}
						break;
					case 2:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + times_symmetrical);
							map.set(j, i, 0, "S" + (-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + (-times_symmetrical));
							map.set(j, i, 0, "S" + times_symmetrical);
							++times_symmetrical;
						}
						break;
					case 3:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + times_symmetrical);
							map.set(i, kernal.rows - 1 - j, 0, "S" + (-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + (-times_symmetrical));
							map.set(i, kernal.rows - 1 - j, 0, "S" + times_symmetrical);
							++times_symmetrical;
						}
						break;
					}
				}
			}
		}
	}

	else if (symmetry == 4)
	{
		for (int j = 0; j < kernal.rows; ++j)
		{
			for (int i = 0; i < kernal.cols; ++i)
			{
				if (kernal.at(i, j, 0) == 0)
				{
					map.set(i, j, 0, "U" + unknown_constants);
					++unknown_constants;
					break;
				}

				else
				{
					if (pos_top)
					{
						map.set(i, j, 0, "S" + times_symmetrical);
						map.set(kernal.cols - 1 - i, j, 0, "S" + (-times_symmetrical));
						++times_symmetrical;
					}

					else
					{
						map.set(i, j, 0, "S" + (-times_symmetrical));
						map.set(kernal.cols - 1 - i, j, 0, "S" + times_symmetrical);
						++times_symmetrical;
					}
				}
			}
		}
	}

	//find all non perfectly canceled cells
	for (int i = N; i < codes.cols - N; ++i)
	{
		for (int j = M; j < codes.rows - M; ++j)
		{
			if (input_matrix.at(i, j, 0) > 0)
			{
				float difference_multiplier = input_matrix.at(i, j, 0);
				for (int x = 0; x < map.cols; ++x)
					for (int y = 0; y < map.rows; ++y)
						if (map.at(x, y, 0).substr(0, 1) == "S")
							difference_multiplier /= abs(kernal.at(x, y, 0));
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				{
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
					{
						if (codes.at(i2, j2, 0).substr(0, 1) != "S" || codes.at(i2, j2, 0).find("*") == std::string::npos)
							//overwrite non multipliers
							codes.set(i2, j2, 0, map.at(map.cols - (i2 - map.cols), map.rows - (j2 - map.rows), 0)
							+ "*" + std::to_string(difference_multiplier));
						else if (codes.at(i2, j2, 0).find("*") != std::string::npos)
						{
							//merge multipliers
							float current_multiplier = std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()));
							std::string new_multiplier = std::to_string((current_multiplier + difference_multiplier) / 2);
							codes.set(i2, j2, 0, (codes.at(i2, j2, 0).replace(codes.at(i2, j2, 0).find("*") + 1, new_multiplier.length(), new_multiplier)));
						}
					}
				}
			}

			else if (input_matrix.at(i, j, 0) < 0)
			{
				float difference_multiplier = input_matrix.at(i, j, 0);
				for (int x = 0; x < map.cols; ++x)
					for (int y = 0; y < map.rows; ++y)
						if (map.at(x, y, 0).substr(0, 2) == "S-")
							difference_multiplier /= abs(kernal.at(x, y, 0));
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				{
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
					{
						if (codes.at(i2, j2, 0).substr(0, 1) != "S" || codes.at(i2, j2, 0).find("*") == std::string::npos
							&& input_matrix.at(i2 - N, j2 - N, 0) != 0)
							codes.set(i2, j2, 0, map.at(map.cols - (i2 - map.cols), map.rows - (j2 - map.rows), 0)
							+ "*" + std::to_string(difference_multiplier));
						else if (codes.at(i2, j2, 0).find("*") != std::string::npos && input_matrix.at(i2 - N, j2 - N, 0) != 0)
						{
							//merge multipliers
							float current_multiplier = std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()));
							std::string new_multiplier = std::to_string((current_multiplier + difference_multiplier) / 2);
							codes.set(i2, j2, 0, (codes.at(i2, j2, 0).replace(codes.at(i2, j2, 0).find("*") + 1, new_multiplier.length(), new_multiplier)));
						}
					}
				}
			}
		}
	}

	//find all perfectly canceled cells
	for (int i = N; i < codes.cols - N; ++i)
	{
		for (int j = M; j < codes.rows - M; ++j)
		{
			if (input_matrix.at(i, j, 0) == 0)
			{
				//map for multipliers for each
				std::map<std::string, float> multipliers;
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
						if (kernal.at(kernal.cols - (i2 - kernal.cols), kernal.rows - (j2 - kernal.rows), 0) != 0)
							if (multipliers.find(codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1)) == multipliers.end())
								multipliers.insert(std::pair<std::string, float>(codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1),
								std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()))));
				//merge symmetric
				for (int i2 = 0; i2 < times_symmetrical; ++i2)
				{
					float new_multiplier = (multipliers["S" + i2] + multipliers["S" + (-i2)]) / 2;
					multipliers["S" + i2] = new_multiplier;
					multipliers["S" + (-i2)] = new_multiplier;
				}
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
						if (kernal.at(kernal.cols - (i2 - kernal.cols), kernal.rows - (j2 - kernal.rows), 0) != 0)
							codes.set(i2, j2, 0, codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1) + "*"
							+ std::to_string(multipliers[codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1)]));
			}
		}
	}

	matrix<float> result(codes.cols, codes.rows, 0);

	//convert to vectors
	std::vector<float> kernal_vector;
	for (int i = 0; i < kernal.cols; ++i)
		for (int j = 0; j < kernal.rows; ++j)
			if (kernal.at(i, j, 0) != 0)
				kernal_vector.push_back(kernal.at(i, j, 0));

	std::vector<float> outputs;
	for (int i = 0; i < input_matrix.cols; ++i)
		for (int j = 0; j < input_matrix.rows; ++j)
			outputs.push_back(input_matrix.at(i, j, 0));

	for (int i = 0; i < codes.cols; i += map.cols)
	{
		for (int j = 0; j < codes.rows; j += map.rows)
		{
			std::vector<float> multipliers_vector(kernal_vector.size());
			for (int i2 = i; i2 < i + map.cols; ++i2)
				for (int j2 = j; j2 < j + map.rows; ++j2)
					multipliers_vector.push_back(std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1,
					codes.at(i2, j2, 0).length())));

			float current_output = outputs[((i / map.cols) * i) + (j / map.rows)];

			//find dot product
			float dot = 0.0f;
			for (int n = 0; n < kernal_vector.size(); ++n)
				dot += (kernal_vector[n] * multipliers_vector[n]);

			//substitute in value
			for (int i2 = i; i2 < i + map.cols; ++i2)
				for (int j2 = j; j2 < j + map.rows; ++j2)
					result.set(i2, j2, 0, current_output / dot);
		}
	}

	return result;
}

matrix<float> convolutional_neural_network::feed_forward(layer input_layer, int num_output)
{
	matrix<float> result(num_output, 1, 1);

	for (int i = 0; i < num_output; ++i)
	{
		//Find sum of weights and run through ReL
		float sum = 0.0f;
		for (int j = 0; j < input_layer.at(0).cols; ++j)
			sum += (input_layer.neuron_at(0, j, 0, 0) * input_layer.data_value_at(i, j, 0, 0));
		result.set(i, 0, 0, max(0, sum));
	}
	return result;
}

matrix<float> convolutional_neural_network::maxpool(matrix<float> input_matrix, int cols, int rows)
{
	std::vector<std::vector<matrix<float>>> samples;
	int across = input_matrix.cols / cols;
	int down = input_matrix.rows / rows;

	//get samples
	for (int i = 0; i < cols; ++i)
	{
		samples.push_back(std::vector<matrix<float>>());
		for (int j = 0; j < rows; ++j)
			samples[i].push_back(input_matrix.from(i * across, j * down, across, down));
	}
	
	//cycle through each sample
	matrix<float> result(cols, rows, 1);
	for (int i = 0; i < samples.size(); ++i)
	{
		for (int j = 0; j < samples[i].size(); ++j)
		{
			//cycle through sample and find max
			float max_value = 0.0f;
			for (int x = 0; x < samples[i][j].cols; ++x)
				for (int y = 0; y < samples[i][j].rows; ++y)
					max_value = max(max_value, samples[i][j].at(x, y, 0));
			result.set(i, j, 0, max_value);
		}
	}
	return result;
}

layer convolutional_neural_network::discriminate_to(int i)
{
	layer current = m_layers[0];
	for (int j = 0; j < i; ++j)
	{
		std::vector<matrix<float>> feature_maps;
		switch (current.type)
		{
		case CNN_CONVOLUTION:
			for (int i2 = 0; i2 = current.feature_map_count; ++i2)
			{
				matrix<float> total = convolve(current.at(i2), current.data_at(0));
				for (int k = 1; k < current.data_count; ++k)
					total += convolve(current.at(i2), current.data_at(k));
				feature_maps.push_back(total);
			}
			current = m_layers[j + 1];
			current.set_feature_maps(feature_maps);
			break;
		case CNN_FEED_FORWARD:
			feature_maps.push_back(feed_forward(current, m_layers[j + 1].at(0).cols));
			current = m_layers[j + 1];
			current.set_feature_maps(feature_maps);
			break;
		case CNN_OUTPUT:
			current.set_feature_maps({ logistic_regression(current.at(0)) });
			return current;
			break;
		}
	}
	return current;
}

layer convolutional_neural_network::generate_until(int i, matrix<float> input_matrix)
{
	return layer();
}

float convolutional_neural_network::max(float a, float b)
{
	return (a > b) ? a : b;
}

matrix<float> convolutional_neural_network::logistic_regression(matrix<float> input_data)
{
	matrix<float> result(input_data.cols, 1, 1);

	float sum = 0.0f;
	for (int i = 0; i < input_data.cols; ++i)
		sum += exp(input_data.at(i, 0, 0));

	for (int i = 0; i < input_data.cols; ++i)
		result.set(i, 0, 0, (exp(input_data.at(i, 0, 0) / sum)));
	return result;
}