#pragma once

#include <vector>
#include <iterator>
#include "Synapse.h"
#include "SimpleNeuron.h"

#define e 2.71828

class Neuron : public SimpleNeuron
{
public:
	Neuron();
	Neuron(std::vector<Synapse>, std::vector<Synapse>);
	~Neuron();
	void FireSynapse();
	void FireInverseSynapse();
	void AddParentOfSynapse(SimpleNeuron);
	void AddChildOfSynapse(SimpleNeuron);
	void IncrementParentWeights(float);
	void DecrementChildWeights(float);
	std::vector<Synapse> GetParentOfSynapses();
	Synapse GetParentOfSynapseAt(int);
	std::vector<Synapse> GetChildOfSynapses();
	Synapse GetChildOfSynapseAt(int);
	Neuron operator-(Neuron);
private:
	std::vector<Synapse> m_ParentOfSynapses;
	std::vector<Synapse> m_ChildOfSynapses;
};

