#pragma once

#include "Neuron.h"
#include "Synapse.h"

class Layer
{
public:
	Layer();
	Layer(std::vector<Neuron> neurons);
	~Layer();
	std::vector<Neuron> GetNeurons();
	Neuron GetNeuronAt(int index);
	void FireNeuronAt(int index, float sum);
	void FireInverseNeuronAt(int index, float sum);
	void IncrementParentWeightAt(int index, float amount);
	void AddNeuron(Neuron neuron);
	Layer operator-(Layer other);
	bool operator==(Layer other);
private:
	std::vector<Neuron> m_Neurons;
};

