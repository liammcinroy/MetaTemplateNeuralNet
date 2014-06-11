#include "Neuron.h"


Neuron::Neuron()
{
}

Neuron::Neuron(Synapse parentOf)
{
	m_ParentOfSynapse = parentOf;
}

Neuron::~Neuron()
{
}

float Neuron::GetValue()
{
	return m_Value;
}

void Neuron::SetValue(float newValue)
{
	m_Value = newValue;
}

float Neuron::FireSynapse(float sum)
{
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

float Neuron::FireInverseSynapse(float sum)
{
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

Synapse Neuron::GetParentOfSynapse()
{
	return m_ParentOfSynapse;
}

void Neuron::IncrementParentWeight(float amount)
{
	m_ParentOfSynapse.SetWeightDiscriminate(m_ParentOfSynapse.GetWeightDiscriminate() + amount);
}

Neuron Neuron::operator-(Neuron other)
{
	SetValue(abs(GetValue() - other.GetValue()));
	return *this;
}