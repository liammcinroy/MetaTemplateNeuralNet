#pragma once
class SimpleNeuron
{
public:
	SimpleNeuron();
	SimpleNeuron(int, int);
	~SimpleNeuron();
	float GetValue();
	int GetLayer();
	int GetIndex();
	void SetValue(float);
private:
	float m_Output;
	int m_Layer;
	int m_Index;
};

