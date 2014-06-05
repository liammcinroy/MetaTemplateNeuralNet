#pragma once

#include "SimpleNeuron.h"
#include <random>

class Synapse
{
public:
	Synapse();
	Synapse(SimpleNeuron parent, SimpleNeuron child);
	~Synapse();
	SimpleNeuron GetParent();
	SimpleNeuron GetChild();
	float GetWeightDiscriminate();
	void SetWeightDiscriminate(float newValue);
	float GetWeightGenerative();
	void SetWeightGenerative(float newValue);
private:
	SimpleNeuron m_Parent;
	SimpleNeuron m_Child;
	float m_WeightDiscriminate;
	float m_WeightGenerate;
};

