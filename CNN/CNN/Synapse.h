#pragma once

#include <random>

class Synapse
{
public:
	Synapse();
	Synapse(int parentLayer, int parentIndex, int startChildIndex, int endChildIndex);
	~Synapse();
	int GetParentIndex();
	int GetParentLayer();
	int GetStartChildIndex();
	int GetEndChildIndex();
	std::vector<int> GetChildrenIndexes();
	float GetWeightDiscriminate();
	void SetWeightDiscriminate(float newValue);
	float GetWeightGenerative();
	void SetWeightGenerative(float newValue);
private:
	int m_ParentIndex;
	int m_ParentLayer;
	int m_StartChildIndex;
	int m_EndChildIndex;
	float m_WeightDiscriminate;
	float m_WeightGenerate;
};

