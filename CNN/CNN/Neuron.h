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
	Neuron(std::vector<Synapse> parentOf, std::vector<Synapse> childOf);
	~Neuron();
	float FireSynapse();
	float FireInverseSynapse();
	void AddParentOfSynapse(SimpleNeuron child);
	void AddChildOfSynapse(SimpleNeuron parent);
	void IncrementParentWeights(float amount);
	void DecrementChildWeights(float amount);
	std::vector<Synapse> GetParentOfSynapses();
	Synapse GetParentOfSynapseAt(int index);
	std::vector<Synapse> GetChildOfSynapses();
	Synapse GetChildOfSynapseAt(int index);
	Neuron operator-(Neuron);
private:
	std::vector<Synapse> m_ParentOfSynapses;
	std::vector<Synapse> m_ChildOfSynapses;
};

