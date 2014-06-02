#include "Neuron.h"


Neuron::Neuron()
{
}

Neuron::Neuron(std::vector<Synapse> parentOf, std::vector<Synapse> childOf)
{
	m_ParentOfSynapses = parentOf;
	m_ChildOfSynapses = childOf;
}

Neuron::~Neuron()
{
}

float Neuron::FireSynapse()
{
	float sum = 0.0f;

	for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.begin(); it != m_ChildOfSynapses.end(); ++it)
		sum += ((*it).GetWeightDiscriminate() * (*it).GetParent().GetValue());

	float probability = (1 / (1 + pow(e, -sum)));

	if (probability > 0.9f)
		return 1.0f;

	else if (probability < 0.1f)
		return 0.0f;

	else
	{
		float random = ((rand() % 100) / 100);
		if (random <= probability)
			return 1.0f;
		else
			return 0.0f;
	}
}

float Neuron::FireInverseSynapse()
{
	float sum = 0.0f;

	for (unsigned int i = m_ParentOfSynapses.size() - 1; i > 0; --i)
		sum += (m_ParentOfSynapses[i].GetWeightGenerative() * m_ParentOfSynapses[i].GetChild().GetValue());

	float probability = -log((1 / sum) - 1);

	if (probability > 0.9f)
		return 1.0f;

	else if (probability < 0.1f)
		return 0.0f;

	else
	{
		float random = ((rand() % 100) / 100);
		if (random <= probability)
			return 1.0f;
		else
			return 0.0f;
	}
}

void Neuron::AddParentOfSynapse(SimpleNeuron child)
{
	m_ParentOfSynapses.push_back(Synapse(SimpleNeuron(GetLayer(), GetIndex()), child));
}

void Neuron::AddChildOfSynapse(SimpleNeuron parent)
{
	m_ChildOfSynapses.push_back(Synapse(SimpleNeuron(GetLayer(), GetIndex()), parent));
}

std::vector<Synapse> Neuron::GetParentOfSynapses()
{
	return m_ParentOfSynapses;
}

Synapse Neuron::GetParentOfSynapseAt(int index)
{
	return m_ParentOfSynapses[index - 1];
}

std::vector<Synapse> Neuron::GetChildOfSynapses()
{
	return m_ChildOfSynapses;
}

Synapse Neuron::GetChildOfSynapseAt(int index)
{
	return m_ChildOfSynapses[index - 1];
}

void Neuron::IncrementParentWeights(float amount)
{
	for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.begin(); it != m_ParentOfSynapses.end(); ++it)
		(*it).SetWeightDiscriminate((*it).GetWeightDiscriminate() + amount);
}

void Neuron::DecrementChildWeights(float amount)
{
	for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.begin(); it != m_ChildOfSynapses.end(); ++it)
		(*it).SetWeightGenerative((*it).GetWeightGenerative() + amount);
}

Neuron Neuron::operator-(Neuron other)
{
	SetValue(abs(GetValue() - other.GetValue()));
	return *this;
}