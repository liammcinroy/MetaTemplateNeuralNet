#include "Synapse.h"


Synapse::Synapse()
{
}

Synapse::Synapse(SimpleNeuron parent, SimpleNeuron child)
{
	m_Parent = parent;
	m_Child = child;
	m_WeightDiscriminate = rand() % 10;
	m_WeightGenerate = rand() % 10;
}

Synapse::~Synapse()
{
}

SimpleNeuron Synapse::GetParent()
{
	return m_Parent;
}

SimpleNeuron Synapse::GetChild()
{
	return m_Child;
}

float Synapse::GetWeightDiscriminate()
{
	return m_WeightDiscriminate;
}

void Synapse::SetWeightDiscriminate(float newOutput)
{
	m_WeightDiscriminate = newOutput;
}

float Synapse::GetWeightGenerative()
{
	return m_WeightGenerate;
}

void Synapse::SetWeightGenerative(float newOutput)
{
	m_WeightGenerate = newOutput;
}