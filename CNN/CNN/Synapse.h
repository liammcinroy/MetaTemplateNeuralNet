#pragma once

#include "SimpleNeuron.h"
#include <random>

class Synapse
{
public:
	Synapse();
	Synapse(SimpleNeuron, SimpleNeuron);
	~Synapse();
	SimpleNeuron GetParent();
	SimpleNeuron GetChild();
	float GetWeightDiscriminate();
	void SetWeightDiscriminate(float);
	float GetWeightGenerative();
	void SetWeightGenerative(float);
private:
	SimpleNeuron m_Parent;
	SimpleNeuron m_Child;
	float m_WeightDiscriminate;
	float m_WeightGenerate;
};

