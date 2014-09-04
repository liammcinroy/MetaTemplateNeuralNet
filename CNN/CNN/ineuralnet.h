#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

template<class T> struct INeuralNet
{
public:
	//save learned net
	virtual void save(const std::string &path);
	//load previously learned net
	virtual void load(const std::string &path);
	//feed forwards
	ILayer<T>* discriminate()
	{
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			Matrix2D<T>* features = layers[i].feed_forwards();
			if (i + 1 < layers.size())
				layers[i + 1].feature_maps = features;
			layers[i + 1].find_probability();
		}
		return layers[layers.size() - 1];
	}
	//set input (batch will not be generated)
	void set_input(Matrix2D<T> input)
	{
		layers[0].feature_maps = input;
	}
	//set labels for batch
	void set_labels(Matrix2D<T> batch_labels)
	{
		labels = batch_labels;
	}
	//wake-sleep algorithm
	void pretrain()
	{
		for (int i = 0; i < layers.size() - 2; ++i)
		{
			if (__typeof(layers[i]) != MaxpoolLayer<T> && __typeof(layers[i + 1]) != MaxpoolLayer<T>)
				layers[i].wake_sleep(layers[i + 1], learning_rate);
		}
	}
	//backpropogate
	void train();
	float learning_rate;
	bool use_dropout;
private:
	std::vector<ILayer> layers;
	Matrix2D<T> labels;
};

template<class T> class FFNeuralNet : INeuralNet<T>
{
public:
	virtual void save(const std::string &path)
	{

	}
	virtual void load(const std::string &path);
};

template<class T> class ConvNeuralNet : INeuralNet<T>
{
public:
	virtual void save(const std::string &path);
	virtual void load(const std::string &path);
};