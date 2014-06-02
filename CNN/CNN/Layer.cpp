#include "Layer.h"


Layer::Layer()
{
}

Layer::Layer(std::vector<Neuron> neurons)
{
	m_Neurons = neurons;
}

Layer::~Layer()
{
}

std::vector<Neuron> Layer::GetNeurons()
{
	return m_Neurons;
}

Neuron Layer::GetNeuronAt(int index)
{
	return m_Neurons[index - 1];
}

float Layer::FireNeuronAt(int index)
{
	return m_Neurons[index - 1].FireSynapse();
}

float Layer::FireInverseNeuronAt(int index)
{
	return m_Neurons[index - 1].FireInverseSynapse();
}

void Layer::AddNeuron(Neuron newNeuron)
{
	m_Neurons.push_back(newNeuron);
}

Layer Layer::operator-(Layer other)
{
	for (int i = 1; i < other.GetNeurons().size(); ++i)
		m_Neurons[i - 1] = m_Neurons[i - 1] - other.GetNeuronAt(i);
	return *this;
}

bool Layer::operator==(Layer other)
{
	for (int i = 1; i < other.GetNeurons().size(); ++i)
		if (m_Neurons[i - 1].GetValue() != other.GetNeuronAt(i).GetValue())
			return false;
	return true;
}