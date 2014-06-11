#include "Synapse.h"


Synapse::Synapse()
{
}

Synapse::Synapse(int parentLayer, int parentIndex, int startChildIndex, int endChildIndex)
{
	m_ParentLayer = parentLayer;
	m_ParentIndex = parentIndex;
	m_StartChildIndex = startChildIndex;
	m_EndChildIndex = endChildIndex;
	m_WeightDiscriminate = rand() % 10;
	m_WeightGenerate = rand() % 10;
}

Synapse::~Synapse()
{
}

int Synapse::GetParentLayer()
{
	return m_ParentLayer;
}

int Synapse::GetParentIndex()
{
	return m_ParentIndex;
}

int Synapse::GetStartChildIndex()
{
	return m_StartChildIndex;
}

int Synapse::GetEndChildIndex()
{
	return m_EndChildIndex;
}

std::vector<int> Synapse::GetChildrenIndexes()
{
	std::vector<int> neurons;
	for (int i = GetStartChildIndex(); i <= GetEndChildIndex(); ++i)
		neurons.push_back(i);
	return neurons;
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