#pragma once

#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

class Layer
{
public:
	Layer();
	Layer(std::vector<Neuron> neurons);
	~Layer();
	std::vector<Neuron> GetNeurons();
	Neuron GetNeuronAt(int index);
	float FireNeuronAt(int index);
	float FireInverseNeuronAt(int index);
	void AddNeuron(Neuron neuron);
	Layer operator-(Layer other);
	bool operator==(Layer other);
private:
	std::vector<Neuron> m_Neurons;
};

