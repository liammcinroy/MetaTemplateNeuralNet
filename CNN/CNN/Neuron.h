#pragma once

#include <vector>
#include <iterator>
#include "Synapse.h"

#define e 2.71828

class Neuron
{
public:
	Neuron();
	Neuron(Synapse parentOf);
	~Neuron();
	float GetValue();
	void SetValue(float);
	float FireSynapse(float sum);
	float FireInverseSynapse(float sum);
	void IncrementParentWeight(float amount);
	Synapse GetParentOfSynapse();
	Neuron operator-(Neuron);
	friend bool operator<(Neuron first, Neuron other)
	{
		return first.m_Value < other.m_Value;
	}
private:
	float m_Value;
	Synapse m_ParentOfSynapse;
};

