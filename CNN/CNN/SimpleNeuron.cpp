#include "SimpleNeuron.h"


SimpleNeuron::SimpleNeuron()
{
}

SimpleNeuron::SimpleNeuron(int layer, int index)
{
	m_Layer = layer;
	m_Index = index;
	m_Output = 0.0f;
}

SimpleNeuron::~SimpleNeuron()
{
}

float SimpleNeuron::GetValue()
{
	return m_Output;
}

void SimpleNeuron::SetValue(float newOutput)
{
	m_Output = newOutput;
}

int SimpleNeuron::GetIndex()
{
	return m_Index;
}

int SimpleNeuron::GetLayer()
{
	return m_Layer;
}
