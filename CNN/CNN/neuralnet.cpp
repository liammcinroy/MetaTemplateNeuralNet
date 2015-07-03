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
	for (int i = 0; i < weight_gradient.size(); ++i)
		for (int j = 0; j < weight_gradient[i].size(); ++j)
			delete weight_gradient[i][j];
	for (int i = 0; i < bias_gradient.size(); ++i)
		for (int j = 0; j < bias_gradient[i].size(); ++j)
			delete bias_gradient[i][j];
}

void NeuralNet::setup_gradient()
{
	//labels and input
	input = std::vector<IMatrix<float>*>(layers[0]->feature_maps.size());
	for (int f = 0; f < input.size(); ++f)
		input[f] = layers[0]->feature_maps[f]->clone();
	labels = std::vector<IMatrix<float>*>(layers[layers.size() - 1]->feature_maps.size());
	for (int f = 0; f < labels.size(); ++f)
		labels[f] = layers[layers.size() - 1]->feature_maps[f]->clone();

	//gradients setup for weights and biases
	weight_gradient = std::vector<std::vector<IMatrix<float>*>>(layers.size());
	bias_gradient = std::vector<std::vector<IMatrix<float>*>>(layers.size());
	if (use_momentum)
	{
		weight_momentum = std::vector<std::vector<IMatrix<float>*>>(layers.size());
		bias_momentum = std::vector<std::vector<IMatrix<float>*>>(layers.size());
	}

	for (int l = 0; l < layers.size(); ++l)
	{
		weight_gradient[l] = std::vector<IMatrix<float>*>(layers[l]->recognition_data.size());
		for (int d = 0; d < layers[l]->recognition_data.size(); ++d)
			weight_gradient[l][d] = layers[l]->recognition_data[d]->clone();

		if (layers[l]->use_biases)
		{
			bias_gradient[l] = std::vector<IMatrix<float>*>(layers[l]->biases.size());
			for (int f_0 = 0; f_0 < layers[l]->biases.size(); ++f_0)
				bias_gradient[l][f_0] = layers[l]->biases[f_0]->clone();
		}

		if (use_momentum)
		{
			weight_momentum[l] = std::vector<IMatrix<float>*>(layers[l]->recognition_data.size());
			for (int d = 0; d < layers[l]->recognition_data.size(); ++d)
				weight_momentum[l][d] = layers[l]->recognition_data[d]->clone();

			if (layers[l]->use_biases)
			{
				bias_momentum[l] = std::vector<IMatrix<float>*>(layers[l]->biases.size());
				for (int f_0 = 0; f_0 < layers[l]->biases.size(); ++f_0)
					bias_momentum[l][f_0] = layers[l]->biases[f_0]->clone();
			}
		}
	}

	//reset gradients
	for (int l = 0; l < weight_gradient.size(); ++l)
		for (int d = 0; d < weight_gradient[l].size(); ++d)
			for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					weight_gradient[l][d]->at(i, j) = 0;
	for (int l = 0; l < bias_gradient.size(); ++l)
		for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
					bias_gradient[l][f_0]->at(i_0, j_0) = 0;
	for (int l = 0; l < weight_momentum.size(); ++l)
		for (int d = 0; d < weight_momentum[l].size(); ++d)
			for (int i = 0; i < weight_momentum[l][d]->rows(); ++i)
				for (int j = 0; j < weight_momentum[l][d]->cols(); ++j)
					weight_momentum[l][d]->at(i, j) = 0;
	for (int l = 0; l < bias_momentum.size(); ++l)
		for (int f_0 = 0; f_0 < bias_momentum[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < bias_momentum[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < bias_momentum[l][f_0]->cols(); ++j_0)
					bias_momentum[l][f_0]->at(i_0, j_0) = 0;
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

void NeuralNet::set_input(std::vector<IMatrix<float>*> &in)
{
	for (int f = 0; f < input.size(); ++f)
	{
		for (int i = 0; i < input[f]->rows(); ++i)
		{
			for (int j = 0; j < input[f]->cols(); ++j)
			{
				input[f]->at(i, j) = in[f]->at(i, j);
				layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);
			}
		}
	}
}

void NeuralNet::set_labels(std::vector<IMatrix<float>*> &batch_labels)
{
	for (int f = 0; f < labels.size(); ++f)
		for (int i = 0; i < labels[f]->rows(); ++i)
			for (int j = 0; j < labels[f]->cols(); ++j)
				labels[f]->at(i, j) = batch_labels[f]->at(i, j);
}

void NeuralNet::discriminate()
{
	//reset
	for (int l = 1; l < layers.size(); ++l)
		for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
			for (int i = 0; i < layers[l]->feature_maps[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->feature_maps[f]->cols(); ++j)
					layers[l]->feature_maps[f]->at(i, j) = 0;
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		if (use_dropout && i != 0 && layers[i]->type != CNN_SOFTMAX)
			dropout(layers[i]);
		layers[i]->feed_forwards(layers[i + 1]->feature_maps);
	}
}

void NeuralNet::pretrain(int iterations)
{
	for (int e = 0; e < iterations; ++e)
	{
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i]->feed_forwards(layers[i + 1]->feature_maps);

			if (layers[i]->type == CNN_CONVOLUTION)
				layers[i]->wake_sleep(learning_rate, use_dropout);
		}
	}
}

float NeuralNet::train(int iterations)
{
	float error;
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();
		error = global_error();

		//values of the network when fed forward
		if (error > .01f)
		{
			std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
			for (int l = 0; l < layers.size(); ++l)
			{
				temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
				for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
					temp[l][f] = layers[l]->feature_maps[f]->clone();
			}

			//get error signals for output and returns any layers to be skipped
			int off = error_signals();

			//backprop for each layer
			if (use_batch_learning)
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weight_gradient[l], bias_gradient[l], learning_rate);
			else if (!use_momentum)
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, layers[l]->recognition_data, layers[l]->biases, learning_rate);
			else
			{
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weight_gradient[l], bias_gradient[l], learning_rate);
				apply_gradient();
			}


			for (int i = 0; i < temp.size(); ++i)
				for (int j = 0; j < temp[i].size(); ++j)
					delete temp[i][j];
		}
	}
	return error;
}

float NeuralNet::train(int iterations, std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases)
{
	float error;
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();
		error = global_error();

		if (error > .01f)
		{
			//values of the network when fed forward
			std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
			for (int l = 0; l < layers.size(); ++l)
			{
				temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
				for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
					temp[l][f] = layers[l]->feature_maps[f]->clone();
			}

			//get error signals for output and returns any layers to be skipped
			int off = error_signals();

			//backprop for each layer
			for (int l = layers.size() - 2 - off; l > 0; --l)
				layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weights[l], biases[l], learning_rate);

			//update weights if applicable
			if (!use_batch_learning)
				apply_gradient(weights, biases);

			for (int i = 0; i < temp.size(); ++i)
				for (int j = 0; j < temp[i].size(); ++j)
					delete temp[i][j];
		}
	}
	return error;
}

void NeuralNet::add_layer(ILayer* layer)
{
	layers.push_back(layer);
}

void NeuralNet::apply_gradient()
{
	if (use_momentum)
	{
		//update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
			for (int d = 0; d < weight_gradient[l].size(); ++d)
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
						layers[l]->recognition_data[d]->at(i, j) += weight_gradient[l][d]->at(i, j) 
																	+ momentum_term * weight_momentum[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += bias_gradient[l][f_0]->at(i_0, j_0)
																+ momentum_term * bias_momentum[l][f_0]->at(i_0, j_0);
	}

	else
	{
		//update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
			for (int d = 0; d < weight_gradient[l].size(); ++d)
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
						layers[l]->recognition_data[d]->at(i, j) += weight_gradient[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += bias_gradient[l][f_0]->at(i_0, j_0);
	}

	//reset gradients
	if (use_momentum)
	{
		for (int l = 0; l < weight_gradient.size(); ++l)
			for (int d = 0; d < weight_gradient[l].size(); ++d)
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
						weight_momentum[l][d]->at(i, j) = momentum_term * weight_momentum[l][d]->at(i, j) + weight_gradient[l][d]->at(i, j);
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						bias_momentum[l][f_0]->at(i_0, j_0) = momentum_term * bias_momentum[l][f_0]->at(i_0, j_0) + bias_gradient[l][f_0]->at(i_0, j_0);
	}

	for (int l = 0; l < weight_gradient.size(); ++l)
		for (int d = 0; d < weight_gradient[l].size(); ++d)
			for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					weight_gradient[l][d]->at(i, j) = 0;
	for (int l = 0; l < bias_gradient.size(); ++l)
		for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
					bias_gradient[l][f_0]->at(i_0, j_0) = 0;
}

void NeuralNet::apply_gradient(std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases)
{
	if (use_momentum)
	{
		//update weights
		for (int l = 0; l < weights.size(); ++l)
			for (int d = 0; d < weights[l].size(); ++d)
				for (int i = 0; i < weights[l][d]->rows(); ++i)
					for (int j = 0; j < weights[l][d]->cols(); ++j)
						layers[l]->recognition_data[d]->at(i, j) += weights[l][d]->at(i, j)
						+ momentum_term * weight_momentum[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < biases.size(); ++l)
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += biases[l][f_0]->at(i_0, j_0)
						+ momentum_term * bias_momentum[l][f_0]->at(i_0, j_0);
	}

	else
	{
		//update weights
		for (int l = 0; l < weights.size(); ++l)
			for (int d = 0; d < weights[l].size(); ++d)
				for (int i = 0; i < weights[l][d]->rows(); ++i)
					for (int j = 0; j < weights[l][d]->cols(); ++j)
						layers[l]->recognition_data[d]->at(i, j) += weights[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < biases.size(); ++l)
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += biases[l][f_0]->at(i_0, j_0);
	}

	//reset gradients
	if (use_momentum)
	{
		for (int l = 0; l < weights.size(); ++l)
			for (int d = 0; d < weights[l].size(); ++d)
				for (int i = 0; i < weights[l][d]->rows(); ++i)
					for (int j = 0; j < weights[l][d]->cols(); ++j)
						weight_momentum[l][d]->at(i, j) = momentum_term * weight_momentum[l][d]->at(i, j) + weights[l][d]->at(i, j);
		for (int l = 0; l < biases.size(); ++l)
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
						bias_momentum[l][f_0]->at(i_0, j_0) = momentum_term * bias_momentum[l][f_0]->at(i_0, j_0) + biases[l][f_0]->at(i_0, j_0);
	}

	for (int l = 0; l < weights.size(); ++l)
		for (int d = 0; d < weights[l].size(); ++d)
			for (int i = 0; i < weights[l][d]->rows(); ++i)
				for (int j = 0; j < weights[l][d]->cols(); ++j)
					weights[l][d]->at(i, j) = 0;
	for (int l = 0; l < biases.size(); ++l)
		for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
					biases[l][f_0]->at(i_0, j_0) = 0;
}

float NeuralNet::global_error()
{
	if (cost_function == CNN_QUADRATIC)
	{
		float sum = 0.0f;
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					sum += pow(labels[k]->at(i, j) - layers[layers.size() - 1]->feature_maps[k]->at(i, j), 2);
		return sum / 2;
	}

	else if (cost_function == CNN_CROSS_ENTROPY)
	{
		float sum = 0.0f;
		for (int k = 0; k < labels.size(); ++k)
		{
			for (int i = 0; i < labels[k]->rows(); ++i)
			{
				for (int j = 0; j < labels[k]->cols(); ++j)
				{
					float t = labels[k]->at(i, j);
					float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

					sum += t * log(x) + (1 - t) * log(1 - x);
				}
			}
		}
		return sum;
	}

	else if (cost_function == CNN_LOG_LIKELIHOOD)
	{
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					if (labels[k]->at(i, j) > 0)
						return -log(layers[layers.size() - 1]->feature_maps[k]->at(i, j));
	}
}

int NeuralNet::error_signals()
{
	if (cost_function == CNN_QUADRATIC)
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					layers[layers.size() - 1]->feature_maps[k]->at(i, j) -= labels[k]->at(i, j);
	if (cost_function == CNN_CROSS_ENTROPY)
	{
		for (int k = 0; k < labels.size(); ++k)
		{
			for (int i = 0; i < labels[k]->rows(); ++i)
			{
				for (int j = 0; j < labels[k]->cols(); ++j)
				{
					float t = labels[k]->at(i, j);
					float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

					layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (x - t) / (x * (1 - x));
				}
			}
		}
	}
	if (cost_function == CNN_LOG_LIKELIHOOD)
	{
		if (layers[layers.size() - 2]->type == CNN_SOFTMAX)
		{
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 2]->feature_maps[k]->at(i, j) -= labels[k]->at(i, j);
			return 1;
		}
		else
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (labels[k]->at(i, j) > 0) ? -1 / layers[layers.size() - 1]->feature_maps[k]->at(i, j) : 0;
	}
	return 0;
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