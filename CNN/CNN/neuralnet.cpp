#include "neuralnet.h"

ILayer NeuralNet::discriminate()
{
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		Matrix2D<int>* features = layers[i].feed_forwards();
		if (i + 1 < layers.size())
			layers[i + 1].feature_maps = features;
		layers[i + 1].find_probability();
	}
	return layers[layers.size() - 1];
}

void NeuralNet::set_input(Matrix2D<int>* input)
{
	layers[0].feature_maps = input;
}

void NeuralNet::set_labels(Matrix2D<int> batch_labels)
{
	labels = batch_labels;
}

void NeuralNet::pretrain()
{
	for (int i = 0; i < layers.size() - 2; ++i)
	{
		if (layers[i].type != CNN_MAXPOOL && layers[i + 1].type != CNN_MAXPOOL)
			layers[i].wake_sleep(layers[i + 1], learning_rate);
	}
}