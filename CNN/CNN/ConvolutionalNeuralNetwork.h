#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <map>

#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

class ConvolutionalNeuralNetwork
{
public:
	ConvolutionalNeuralNetwork();
	ConvolutionalNeuralNetwork(std::string path);
	ConvolutionalNeuralNetwork(int* neuronsOnEachLayer, int* featureMapsPerLayer, int* featureMapDimensions, int* featureMapConnections[], int* featureMapStartIndex[] );
	~ConvolutionalNeuralNetwork();
	void AddLayer(Layer newLayers);
	std::vector<Layer> GetLayers();
	Layer GetLayerAt(int index);
	Layer GetInput();
	void SetInput(float* input[], int width, int height);
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

