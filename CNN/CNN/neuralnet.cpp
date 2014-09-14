#pragma once

#include "neuralnet.h"

NeuralNet::NeuralNet()
{
	layers = std::vector<ILayer*>();
	srand(time(NULL)); //LNK2019
}

void NeuralNet::save(std::string path)
{
}

void NeuralNet::load(std::string path)
{
}

ILayer* NeuralNet::discriminate()
{
	for (std::vector<ILayer*>::iterator it = layers.begin(); it + 1 != layers.end(); ++it)
	{
		Matrix2D<int>* features = (*it)->feed_forwards();
		(*(it + 1))->feature_maps = features;
		(*(it + 1))->find_probability();
	}
	return layers[layers.size() - 1];
}

void NeuralNet::set_input(Matrix2D<int>* input)
{
	layers[0]->feature_maps = input;
}

void NeuralNet::set_labels(Matrix2D<int> batch_labels)
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