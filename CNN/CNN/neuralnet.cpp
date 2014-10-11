#pragma once

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

NeuralNet::NeuralNet()
{
	srand(time(NULL)); //LNK2019
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
					file << layers[l]->data[f]->at(i, j) << ',';//data values
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
		std::vector<Matrix<int>*> features = (*it)->feed_forwards();
		(*(it + 1))->feature_maps = features;
		(*(it + 1))->find_probability();
	}
	return layers[layers.size() - 1];
}

void NeuralNet::set_input(std::vector<Matrix<int>*> input)
{
	layers[0]->feature_maps = input;
}

void NeuralNet::set_labels(Matrix<int>* batch_labels)
{
	labels = batch_labels;
}

void NeuralNet::pretrain()
{
	for (int i = 0; i < layers.size() - 2; ++i)
	{
		if (layers[i]->type != CNN_MAXPOOL && layers[i + 1]->type != CNN_MAXPOOL)
			layers[i]->wake_sleep(*layers[i + 1], learning_rate);
	}
}

void NeuralNet::train()
{

}

void NeuralNet::add_layer(ILayer* layer)
{
	layers.push_back(layer);
}