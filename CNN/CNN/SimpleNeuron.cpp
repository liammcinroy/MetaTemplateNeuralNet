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
	delete &m_Output;
	delete &m_Index;
	delete &m_Layer;
}

float SimpleNeuron::GetOutput()
{
	return m_Output;
}

void SimpleNeuron::SetOutput(float newOutput)
{
	m_Output = newOutput;
}

