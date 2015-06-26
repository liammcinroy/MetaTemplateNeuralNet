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

void NeuralNet::train(int iterations)
{
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();

		//values of the network when fed forward
		std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
		for (int l = 0; l < layers.size(); ++l)
		{
			temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
			for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
				temp[l][f] = layers[l]->feature_maps[f]->clone();
		}

		//backprop for each layer
		for (int l = layers.size() - 1; l > 0; --l)
		{
			if (layers[l]->type == CNN_OUTPUT)
				layers[l]->back_prop(labels, std::vector<IMatrix<float>*>(), std::vector<IMatrix<float>*>(), std::vector<IMatrix<float>*>());
			else
				layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weight_gradient[l], bias_gradient[l]);
		}
		
		//update weights if applicable
		if (!use_batch_learning)
			apply_gradient();

		for (int i = 0; i < temp.size(); ++i)
			for (int j = 0; j < temp[i].size(); ++j)
				delete temp[i][j];
	}
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
						layers[l]->recognition_data[d]->at(i, j) += -learning_rate * weight_gradient[l][d]->at(i, j) 
																	+ -momentum_term * weight_momentum[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * bias_gradient[l][f_0]->at(i_0, j_0)
																+ -momentum_term * bias_momentum[l][f_0]->at(i_0, j_0);
	}

	else
	{
		//update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
			for (int d = 0; d < weight_gradient[l].size(); ++d)
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
						layers[l]->recognition_data[d]->at(i, j) += -learning_rate * weight_gradient[l][d]->at(i, j);

		//update biases
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * bias_gradient[l][f_0]->at(i_0, j_0);
	}

	//reset gradients
	if (use_momentum)
	{
		for (int l = 0; l < weight_gradient.size(); ++l)
			for (int d = 0; d < weight_gradient[l].size(); ++d)
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
						weight_momentum[l][d]->at(i, j) = weight_gradient[l][d]->at(i, j);
		for (int l = 0; l < bias_gradient.size(); ++l)
			for (int f_0 = 0; f_0 < bias_gradient[l].size(); ++f_0)
				for (int i_0 = 0; i_0 < bias_gradient[l][f_0]->rows(); ++i_0)
					for (int j_0 = 0; j_0 < bias_gradient[l][f_0]->cols(); ++j_0)
						bias_momentum[l][f_0]->at(i_0, j_0) = bias_gradient[l][f_0]->at(i_0, j_0);
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

float NeuralNet::global_error()
{
	float sum = 0.0f;
	for (int k = 0; k < labels.size(); ++k)
		for (int i = 0; i < labels[k]->rows(); ++i)
			for (int j = 0; j < labels[k]->cols(); ++j)
				sum += pow(labels[k]->at(i, j) - layers[layers.size() - 1]->feature_maps[k]->at(i, j), 2);
	return sum / 2;
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