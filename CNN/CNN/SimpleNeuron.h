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
	friend bool operator<(SimpleNeuron first, SimpleNeuron other)
	{
		if (other.GetLayer() < first.GetLayer())
			return true;

		else if (other.GetLayer() == first.GetLayer())
		{
			if (other.GetIndex() < first.GetIndex())
				return true;
			else
				return false;
		}

		else
			return false;
	}
private:
	float m_Output;
	int m_Layer;
	int m_Index;
};

