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

void Neuron::FireSynapse()
{
	float sum = 0.0f;

	for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.begin(); it != m_ChildOfSynapses.end(); ++it)
		sum += ((*it).GetWeightDiscriminate() * (*it).GetParent().GetOutput());

	float probability = (1 / (1 + pow(e, -sum)));

	if (probability > 0.9f)
		for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.begin(); it != m_ParentOfSynapses.end(); ++it)
			(*it).GetChild().SetOutput(1.0f);

	else if (probability < 0.1f)
		for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.begin(); it != m_ParentOfSynapses.end(); ++it)
			(*it).GetChild().SetOutput(0.0f);

	else
	{
		float random = ((rand() % 100) / 100);
		if (random <= probability)
			for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.begin(); it != m_ParentOfSynapses.end(); ++it)
				(*it).GetChild().SetOutput(1.0f);
		else
			for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.begin(); it != m_ParentOfSynapses.end(); ++it)
				(*it).GetChild().SetOutput(0.0f);
	}
}

void Neuron::FireInverseSynapse()
{
	float sum = 0.0f;

	for (std::vector<Synapse>::iterator it = m_ParentOfSynapses.end(); it != m_ParentOfSynapses.begin(); --it)
		sum += ((*it).GetWeightGenerative() * (*it).GetChild().GetOutput());

	float probability = -log((1 / sum) - 1);

	if (probability > 0.9f)
		for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.end(); it != m_ChildOfSynapses.begin(); --it)
			(*it).GetParent().SetOutput(1.0f);

	else if (probability < 0.1f)
		for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.end(); it != m_ChildOfSynapses.begin(); --it)
			(*it).GetParent().SetOutput(0.0f);

	else
	{
		float random = ((rand() % 100) / 100);
		if (random <= probability)
			for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.end(); it != m_ChildOfSynapses.begin(); ++it)
				(*it).GetParent().SetOutput(1.0f);
		else
			for (std::vector<Synapse>::iterator it = m_ChildOfSynapses.end(); it != m_ChildOfSynapses.begin(); ++it)
				(*it).GetParent().SetOutput(0.0f);
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
	SetOutput(abs(GetOutput() - other.GetOutput()));
	return *this;
}