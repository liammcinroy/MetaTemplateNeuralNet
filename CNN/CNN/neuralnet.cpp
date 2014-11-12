//#pragma once

#include "neuralnet.h"

std::string inbetween(std::string &input, const std::string &start, const std::string &end)
{
	int i = input.find(start) + start.size();
	return input.substr(i, input.substr(i, input.size()).find(end));
}

std::vector<std::string> split(std::string &input, const std::string &delim)
{
	std::vector<std::string> result;
	int last = input.find(delim);
	std::string item = input.substr(0, last);
	if (item == "")
		return result;
	result.push_back(item);

	while (last != std::string::npos && last < input.size())
	{
		try
		{
			item = input.substr(last + delim.size(), input.substr(last + delim.size(), input.size()).find(delim));
		}
		catch (int e)
		{
			item = input.substr(last + delim.size(), input.size());
		}
		if (item != "")
			result.push_back(item);
		last += delim.size() + item.size();
	}
	return result;
}

NeuralNet::~NeuralNet()
{
	for (int i = 0; i < layers.size(); ++i)
		delete layers[i];
}

void NeuralNet::save_data(std::string path)
{
	std::ofstream file(path);
	for (int l = 0; l < layers.size(); ++l)
	{
		file << '[';//begin data values
		for (int f = 0; f < layers[l]->data.size(); ++f)
			for (int i = 0; i < layers[l]->data[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->data[f]->cols(); ++j)
					file << std::to_string(layers[l]->data[f]->at(i, j)) << ',';//data values
		file << ']';//end data values
	}
	file.flush();
}

void NeuralNet::load_data(std::string path)
{
	std::ifstream file(path);
	std::string layer;
	int l = 0;
	while (std::getline(file, layer, '['))
	{
		if (layer == "")
			continue;
		std::vector<std::string> data_s = split(inbetween(layer, "[", "]"), ",");
		int c = 0;
		int i = 0;
		int j = 0;
		for (int k = 0; k < data_s.size(); ++k, ++j)
		{
			if (j >= layers[l]->data[c]->cols())
			{
				j = 0;
				++i;
			}
			if (i >= layers[l]->data[c]->rows())
			{
				i = 0;
				++c;
			}

			layers[l]->data[c]->at(i, j) = std::stof(data_s[k]);
		}
		++l;
	}
}

ILayer* NeuralNet::discriminate()
{
	for (std::vector<ILayer*>::iterator it = layers.begin(); it + 1 != layers.end(); ++it)
	{
		if (binary_net)
			(*(it + 1))->feature_maps = (*it)->feed_forwards_prob();
		else
			(*(it + 1))->feature_maps = (*it)->feed_forwards();
	}
	return layers[layers.size() - 1];
}

void NeuralNet::set_input(std::vector<Matrix<float>*> input)
{
	layers[0]->feature_maps = input;
}

void NeuralNet::set_labels(std::vector<Matrix<float>*> batch_labels)
{
	labels = batch_labels;
}

void NeuralNet::pretrain()
{
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		if (layers[i]->type != CNN_MAXPOOL)
			layers[i]->wake_sleep(learning_rate, binary_net);
		else
		{
			if (binary_net)
				layers[i + 1]->feature_maps = layers[i]->feed_forwards_prob();
			else
				layers[i + 1]->feature_maps = layers[i]->feed_forwards();
		}
	}
}

void NeuralNet::train(int epochs)
{
	for (int e = 0; e < epochs; ++e)
	{
		discriminate();

		//find output's error signals and set the output layer to them for backprop
		int top = layers.size() - 1;
		for (int k = 0; k < layers[top]->feature_maps.size(); ++k)
			for (int i = 0; i < layers[top]->feature_maps[k]->rows(); ++i)
				for (int j = 0; j < layers[top]->feature_maps[k]->cols(); ++j)
					layers[top]->feature_maps[k]->at(i, j) = output_error_signal(i, j, k);

		//feeding all the layers backwards and multiplying by derivative of the sigmoid (or of y=x)
		//after setting the output layer's data to its error signal will give all of the error signals
		for (int l = top; l > 1; --l)
		{
			if (layers[l - 1]->type != CNN_MAXPOOL)
			{
				std::vector<Matrix<float>*> temp = layers[l - 1]->feed_backwards(layers[l]->feature_maps);
				for (int f = 0; f < layers[l - 1]->feature_maps.size(); ++f)
				{
					for (int i = 0; i < layers[l - 1]->feature_maps[f]->rows(); ++i)
					{
						for (int j = 0; j < layers[l - 1]->feature_maps[f]->cols(); ++j)
						{
							float y = layers[l - 1]->feature_maps[f]->at(i, j);
							float delta_k = y;

							if (binary_net)
								delta_k *= y * (1 - y) * temp[f]->at(i, j);
							else
								delta_k *= temp[f]->at(i, j);

							layers[l - 1]->feature_maps[f]->at(i, j) = y * (1 - y) * temp[f]->at(i, j);

							float delta_weight = -learning_rate * delta_k;

							//Update the weights
							if (layers[l - 1]->type == CNN_FEED_FORWARD)
							{
								for (int j2 = 0; j2 < layers[l]->feature_maps[f]->rows(); ++j2)
									layers[l - 1]->data[f]->at(j2, i) += delta_weight * layers[l]->feature_maps[f]->at(j2, 0);
							}

							else if (layers[l - 1]->type == CNN_CONVOLUTION)
							{
								int max_i = layers[l]->feature_maps[f]->rows();
								int max_j = layers[l]->feature_maps[f]->cols();

								int r = (layers[l - 1]->data[f]->rows() - 1) / 2;

								for (int n = r; n >= -r; --n)
								{
									for (int m = r; m >= -r; --m)
									{
										int up_i = i - n;
										int up_j = j - m;

										if (up_i > 0 && up_i < max_i && up_j > 0 && up_j < max_j)
											layers[l - 1]->data[f]->at(r - n, r - m) += delta_weight * layers[l]->feature_maps[f]->at(up_i, up_j);
									}
								}
							}
						}
					}
				}
			}

			else
			{
				for (int f = 0; f < layers[l - 1]->feature_maps.size(); ++f)
				{
					int currentSampleI = 0;
					int currentSampleJ = 0;

					int down = layers[l - 1]->feature_maps[f]->rows() / layers[l + 1]->feature_maps[f]->rows();
					int across = layers[l - 1]->feature_maps[f]->cols() / layers[l + 1]->feature_maps[f]->cols();

					for (int i = 0; i < layers[l - 1]->feature_maps[f]->rows(); ++i)
					{
						for (int j = 0; j < layers[l - 1]->feature_maps[f]->cols(); ++j)
						{
							std::pair<int, int> coords = layers[l - 1]->coords_of_max[f]->at(currentSampleI, currentSampleJ);
							if (i == coords.first && j == coords.second)
							{
								float y = layers[l - 1]->feature_maps[f]->at(i, j);

								if (binary_net)
									layers[l - 1]->feature_maps[f]->at(i, j) = y * (1 - y) * layers[l]->feature_maps[f]->at(currentSampleI, currentSampleJ);
								else
									layers[l - 1]->feature_maps[f]->at(i, j) = layers[l]->feature_maps[f]->at(currentSampleI, currentSampleJ);
							}

							else
								layers[l - 1]->feature_maps[f]->at(i, j) = 0;

							if (currentSampleJ % across == 0)
								++currentSampleJ;
						}
						if (currentSampleI % down == 0)
							++currentSampleI;
					}
				}
			}
		}
	}
}

void NeuralNet::add_layer(ILayer* layer)
{
	layers.push_back(layer);
}

float NeuralNet::global_error()
{
	float sum = 0.0f;
	for (int k = 0; k < labels.size(); ++k)
	for (int i = 0; i < labels[k]->rows(); ++i)
		for (int j = 0; j < labels[k]->cols(); ++j)
			sum += pow(labels[k]->at(i, j) - layers[layers.size() - 1]->feature_maps[0]->at(i, j), 2);
	return sum / 2;
}

float NeuralNet::output_error_signal(int &i, int &j, int &k)
{
	float O = layers[layers.size() - 1]->feature_maps[k]->at(i, j);
	float t = labels[k]->at(i, j);

	if (binary_net)
		return O * (O - t) * O * (1 - O);//delta k
	else
		return  O * (O - t);
}

float NeuralNet::error_signal(int &i, int &j, int &k, float &weights_sum)
{
	float y = layers[k]->feature_maps[0]->at(i, j);
	return y * (y - 1) * weights_sum;
}

Matrix2D<int, 4, 1>* NeuralNet::coords(int &l, int &k, int &i, int &j)
{
	Matrix2D<int, 4, 1>* out = new Matrix2D<int, 4, 1>();
	out->at(0, 0) = l;
	out->at(1, 0) = k;
	out->at(2, 0) = i;
	out->at(3, 0) = j;
	return out;
}
