#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <string>

#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

class ConvolutionalNeuralNetwork
{
public:
	ConvolutionalNeuralNetwork();
	ConvolutionalNeuralNetwork(std::string);
	~ConvolutionalNeuralNetwork();
	void AddLayer(Layer);
	std::vector<Layer> GetLayers();
	Layer GetLayerAt(int);
	Layer GetInput();
	void SetInput(float**, int, int);
	Layer GetOutput();
	void SetOutput(float**, int, int);
	Layer Discriminate();
	Layer Generate(Layer);
	void LearnCurrentInput();
	float GetLearnRate();
	void SetLearnRate(float);
	void ReadFromFile(std::string);
	void SaveToFile(std::string);
private:
	std::vector<Layer> m_Layers;
	float m_LearnRate;
};

