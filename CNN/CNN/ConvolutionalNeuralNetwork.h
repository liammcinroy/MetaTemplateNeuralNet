#pragma once

#include <map>
#include <fstream>
#include <istream>
#include <iostream>
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
	ConvolutionalNeuralNetwork(std::string path);
	ConvolutionalNeuralNetwork(std::vector<int> neuronCountPerLayer, std::vector<int> featureMapsPerLayer, std::vector<int> featureMapDimensions,
		std::vector<std::vector<int>> featureMapConnections, std::vector<std::vector<int>> featureMapStartIndex);
	~ConvolutionalNeuralNetwork();
	void AddLayer(Layer newLayers);
	std::vector<Layer> GetLayers();
	Layer GetLayerAt(int index);
	Layer GetInput();
	void SetInput(std::vector<std::vector<float>> input);
	Layer GetOutput();
	Layer Discriminate();
	Layer Generate(Layer input);
	void LearnCurrentInput();
	float GetLearnRate();
	void SetLearnRate(float newRate);
	void ReadFromFile(std::string path);
	void SaveToFile(std::string path);
private:
	std::vector<Layer> m_Layers;
	float m_LearnRate;
	std::string StringUntil(std::string, std::string);
	std::string StringBy(std::string, std::string);
	std::string FindInBetween(std::string, std::string, std::string);
};

