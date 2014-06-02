#pragma once

#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

class Layer
{
public:
	Layer();
	Layer(std::vector<Neuron>);
	~Layer();
	std::vector<Neuron> GetNeurons();
	Neuron GetNeuronAt(int);
	void FireNeuornAt(int);
	void AddNeuron(Neuron);
	Layer operator-(Layer);
	bool operator==(Layer);
private:
	std::vector<Neuron> m_Neurons;
};

