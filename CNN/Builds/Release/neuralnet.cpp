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
	for (int i = 0; i < input.size(); ++i)
		delete input[i];
	for (int i = 0; i < labels.size(); ++i)
		delete labels[i];
}

void NeuralNet::save_data(std::string path)
{
	std::ofstream file(path);
	for (int l = 0; l < layers.size(); ++l)
	{
		//begin recognition_data values
		for (int f = 0; f < layers[l]->recognition_data.size(); ++f)
			for (int i = 0; i < layers[l]->recognition_data[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->recognition_data[f]->cols(); ++j)
					file << std::to_string(layers[l]->recognition_data[f]->at(i, j)) << ',';//recognition_data values
		//end recognition_data values
	}

	for (int l = 0; l < layers.size(); ++l)
	{
		//begin generative_data values
		for (int f = 0; f < layers[l]->generative_data.size(); ++f)
			for (int i = 0; i < layers[l]->generative_data[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->generative_data[f]->cols(); ++j)
					file << std::to_string(layers[l]->generative_data[f]->at(i, j)) << ',';//recognition_data values
		//end generative_data values
	}

	for (int l = 0; l < layers.size(); ++l)
	{
		//begin biases values
		for (int f = 0; f < layers[l]->biases.size(); ++f)
			for (int i = 0; i < layers[l]->biases[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->biases[f]->cols(); ++j)
					file << std::to_string(layers[l]->biases[f]->at(i, j)) << ',';//recognition_data values
		//end biases values
	}
	file.flush();
}

void NeuralNet::load_data(std::string path)
{
	std::ifstream file(path);
	std::string data;
	int l = 0;
	int c = 0;
	int i = 0;
	int j = 0;
	bool greater = false;
	bool biases = false;
	while (std::getline(file, data, ','))
	{
		float value = std::stof(data);
		if (!greater)
		{
			if (j >= layers[l]->recognition_data[c]->cols())
			{
				j = 0;
				++i;
			}
			if (i >= layers[l]->recognition_data[c]->rows())
			{
				i = 0;
				++c;
			}
			if (c >= layers[l]->recognition_data.size())
			{
				c = 0;
				i = 0;
				j = 0;
				++l;
			}

			layers[l]->recognition_data[c]->at(i, j) = value;
		}

		else if (!biases)
		{
			if (j >= layers[l]->generative_data[c]->cols())
			{
				j = 0;
				++i;
			}
			if (i >= layers[l]->generative_data[c]->rows())
			{
				i = 0;
				++c;
			}
			if (c >= layers[l]->generative_data.size())
			{
				c = 0;
				i = 0;
				j = 0;
				++l;
			}

			layers[l]->generative_data[c]->at(i, j) = value;
		}

		else
		{
			if (j >= layers[l]->biases[c]->cols())
			{
				j = 0;
				++i;
			}
			if (i >= layers[l]->biases[c]->rows())
			{
				i = 0;
				++c;
			}
			if (c >= layers[l]->biases.size())
			{
				c = 0;
				i = 0;
				j = 0;
				++l;
			}

			layers[l]->recognition_data[c]->at(i, j) = value;
		}

		if (l >= layers.size())
		{
			if (!greater)
				greater = true;
			else
				biases = true;
			c = 0;
			i = 0;
			j = 0;
			l = 0;
		}
	}
}

void NeuralNet::set_input(std::vector<IMatrix<float>*> in)
{
	input = in;
	for (int f = 0; f < input.size(); ++f)
		for (int i = 0; i < input[f]->rows(); ++i)
			for (int j = 0; j < input[f]->cols(); ++j)
				layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);
}

void NeuralNet::set_labels(std::vector<IMatrix<float>*> batch_labels)
{
	labels = batch_labels;
}

ILayer* NeuralNet::discriminate()
{
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		if (use_dropout && i != 0 && layers[i]->type != CNN_SOFTMAX)
			dropout(layers[i]);
		if (binary_net && layers[i + 1]->type != CNN_SOFTMAX)
			layers[i]->feed_forwards_prob(layers[i + 1]->feature_maps);
		else
			layers[i]->feed_forwards(layers[i + 1]->feature_maps);
	}
	return layers[layers.size() - 1];
}

void NeuralNet::pretrain(int epochs)
{
	for (int e = 0; e < epochs; ++e)
	{
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			if (binary_net)
				layers[i]->feed_forwards_prob(layers[i + 1]->feature_maps);
			else
				layers[i]->feed_forwards(layers[i + 1]->feature_maps);

			if (layers[i]->type == CNN_CONVOLUTION)
				layers[i]->wake_sleep(learning_rate, binary_net, use_dropout);
		}
	}
}

void NeuralNet::train(int epochs)
{
	for (int e = 0; e < epochs; ++e)
	{
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();

		//find output's error signals and set the output layer to them for backprop
		int top = layers.size() - 1;
		for (int k = 0; k < layers[top]->feature_maps.size(); ++k)
			for (int i = 0; i < layers[top]->feature_maps[k]->rows(); ++i)
				for (int j = 0; j < layers[top]->feature_maps[k]->cols(); ++j)
					layers[top]->feature_maps[k]->at(i, j) = output_error_signal(i, j, k);

		//feeding all the layers backwards and multiplying by derivative of the sigmoid (or of y=x)
		//after setting the output layer's recognition_data to its error signal will give all of the error signals
		for (int l = top; l > 0; --l)
		{
			if (layers[l - 1]->type == CNN_PERCEPTRON)
			{
				std::vector<IMatrix<float>*> temp = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
				for (int i = 0; i < temp.size(); ++i)
					temp[i] = layers[l]->feature_maps[i]->clone();

				layers[l - 1]->feed_backwards(layers[l]->feature_maps, false);
				for (int f_o = 0; f_o < layers[l]->feature_maps.size(); ++f_o)
				{
					for (int i = 0; i < layers[l]->feature_maps[f_o]->rows(); ++i)
					{
						for (int j = 0; j < layers[l]->feature_maps[f_o]->cols(); ++j)
						{
							float y = temp[f_o]->at(i, j);
							float delta_k = 1.0f;

							if (binary_net)
								delta_k *= y * (1 - y);

							float delta_j = -learning_rate * delta_k * layers[l]->feature_maps[f_o]->at(i, j);
							layers[l - 1]->biases[f_o]->at(i, j) += -learning_rate * y * (1 - y);

							//Update the weights
							for (int f = 0; f < layers[l - 1]->feature_maps.size(); ++f)
								for (int i2 = 0; i2 < layers[l - 1]->feature_maps[f]->rows(); ++i2)
									for (int j2 = 0; j2 < layers[l - 1]->feature_maps[f]->cols(); ++j2)
										layers[l - 1]->recognition_data[0]->at(f_o * layers[l]->feature_maps[f_o]->rows() *
										layers[l]->feature_maps[f_o]->cols() + i * layers[l]->feature_maps[f_o]->cols() + j,
										f * layers[l - 1]->feature_maps[f]->rows() * layers[l - 1]->feature_maps[f]->cols() +
										i2 * layers[l - 1]->feature_maps[f]->cols() + j2)
										+= delta_j * layers[l - 1]->feature_maps[f]->at(i2, j2);
						}
					}
				}

				for (int i = 0; i < temp.size(); ++i)
					delete temp[i];
			}

			else if (layers[l - 1]->type == CNN_MAXPOOL)
			{
				for (int f = 0; f < layers[l - 1]->feature_maps.size(); ++f)
				{
					int currentSampleI = 0;
					int currentSampleJ = 0;

					int down = layers[l - 1]->feature_maps[f]->rows() / layers[l]->feature_maps[f]->rows();
					int across = layers[l - 1]->feature_maps[f]->cols() / layers[l]->feature_maps[f]->cols();

					for (int i = 0; i < layers[l - 1]->feature_maps[f]->rows(); ++i)
					{
						for (int j = 0; j < layers[l - 1]->feature_maps[f]->cols(); ++j)
						{
							std::pair<int, int> coords = layers[l - 1]->coords_of_max[f]->at(currentSampleI, currentSampleJ);
							if (i == coords.first && j == coords.second)
							{
								float y = layers[l - 1]->feature_maps[f]->at(i, j);

								if (binary_net)
									layers[l - 1]->feature_maps[f]->at(i, j) = y * (1 - y) * layers[l]->feature_maps[f]->at(i, j);
								else
									layers[l - 1]->feature_maps[f]->at(i, j) = layers[l]->feature_maps[f]->at(i, j);
							}

							else
								layers[l - 1]->feature_maps[f]->at(i, j) = 0;

							if (j % across == 0 && j != 0)
								++currentSampleJ;
						}
						if (i % down == 0 && i != 0)
							++currentSampleI;
					}
				}
			}

			else if (layers[l - 1]->type == CNN_SOFTMAX)
			{
				for (int f = 0; f < layers[l - 1]->feature_maps.size(); ++f)
					for (int i = 0; i < layers[l - 1]->feature_maps[f]->rows(); ++i)
						for (int j = 0; j < layers[l - 1]->feature_maps[f]->cols(); ++j)
							layers[l - 1]->feature_maps[f]->at(i, j) = layers[l]->feature_maps[f]->at(i, j);
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
		return (O - t) * O * (1 - O);//delta k
	else
		return  (O - t);
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

void NeuralNet::dropout(ILayer* &layer)
{
	for (int f = 0; f < layer->feature_maps.size(); ++f)
		for (int i = 0; i < layer->feature_maps[f]->rows(); ++i)
			for (int j = 0; j < layer->feature_maps[f]->cols(); ++j)
				if ((1.0f * rand()) / RAND_MAX >= .5f)
					layer->feature_maps[f]->at(i, j) = 0;
}