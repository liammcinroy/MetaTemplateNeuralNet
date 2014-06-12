#include "ConvolutionalNeuralNetwork.h"

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork()
{
}

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::string path)
{
	ReadFromFile(path);
}

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::vector<int> neuronCountPerLayer, std::vector<int> featureMapsPerLayer, std::vector<int> featureMapDimensions, std::vector<std::vector<int>> featureMapConnections, std::vector<std::vector<int>> featureMapStartIndex)
{
	for (unsigned int i = 0; i < neuronCountPerLayer.size() - 1; ++i)
	{
		Layer currentLayer;

		for (int j = 0; j < neuronCountPerLayer[i]; ++j)
		{
			if (featureMapsPerLayer[i] == 1)
				currentLayer.AddNeuron(Neuron(Synapse(i + 1, j + 1, 1, neuronCountPerLayer[i + 1])));

			else
			{
				int featureMapsUp = featureMapsPerLayer[i + 1];
				int inFeatureMap = featureMapsPerLayer[i] / j;
				int startIndex = (neuronCountPerLayer[i + 1] / featureMapsUp) * featureMapStartIndex[i][inFeatureMap];
				int destinationIndex = startIndex + (neuronCountPerLayer[i + 1] / featureMapsUp) * featureMapConnections[i][inFeatureMap];

				currentLayer.AddNeuron(Neuron(Synapse(i + 1, j + 1,
					(neuronCountPerLayer[i + 1] / featureMapsUp) * featureMapStartIndex[i][inFeatureMap],
					startIndex + (neuronCountPerLayer[i + 1] / featureMapsUp) * featureMapConnections[i][inFeatureMap])));
			}
		}
		AddLayer(currentLayer);
	}

	Layer output;

	for (int i = 0; i < neuronCountPerLayer[neuronCountPerLayer.size() - 1]; ++i)
		output.AddNeuron(Neuron(Synapse()));
	AddLayer(output);
}

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork()
{
}

void ConvolutionalNeuralNetwork::AddLayer(Layer newLayer)
{
	if (m_Layers.size() > 0)
		m_Layers.push_back(newLayer);
	else
	{
		m_Layers.push_back(Layer());
		m_Layers.push_back(newLayer);
	}
}

std::vector<Layer> ConvolutionalNeuralNetwork::GetLayers()
{
	return m_Layers;
}

Layer ConvolutionalNeuralNetwork::GetLayerAt(int index)
{
	return m_Layers[index - 1];
}

Layer ConvolutionalNeuralNetwork::GetInput()
{
	return GetLayerAt(1);
}

void ConvolutionalNeuralNetwork::SetInput(std::vector<std::vector<float>> input)
{
	std::vector<Neuron> neurons;
	for (unsigned int i = 0; i < input.size(); ++i)
		for (unsigned int j = 0; j < input[i].size(); ++j)
			neurons.push_back(Neuron(Synapse(1, i + (j * input[i].size() + 1), 1, m_Layers[1].GetNeurons().size())));
	m_Layers[0] = Layer(neurons);
}

Layer ConvolutionalNeuralNetwork::GetOutput()
{
	return GetLayerAt(m_Layers.size());
}

Layer ConvolutionalNeuralNetwork::DiscriminateUntil(unsigned int index)
{
	for (unsigned int l = 1; l < index; ++l)
	{
		for (unsigned int n = 1; n < m_Layers[l].GetNeurons().size(); ++n)
		{
			float sum = 0.0f;

			for (unsigned int n2 = 1; n2 < m_Layers[l - 1].GetNeurons().size(); ++n2)
			{
				unsigned int startIndex = m_Layers[l - 1].GetNeuronAt(n2).GetParentOfSynapse().GetStartChildIndex();
				unsigned int endIndex = m_Layers[l - 1].GetNeuronAt(n2).GetParentOfSynapse().GetEndChildIndex();
				float weight = m_Layers[l - 1].GetNeuronAt(n2).GetParentOfSynapse().GetWeightGenerative();

				if (startIndex <= n && n <= endIndex)
					sum += (m_Layers[l - 1].GetNeuronAt(n2).GetValue() * weight);
			}
			m_Layers[l].FireNeuronAt(n, sum);
		}
	}

	return GetOutput();
}

Layer ConvolutionalNeuralNetwork::GenerateUntil(Layer input, unsigned int index)
{
	for (unsigned int l = m_Layers.size() - 2; l > index - 1; --l)
	{
		for (unsigned int n = 1; n < m_Layers[l].GetNeurons().size(); ++n)
		{
			unsigned int startIndex = m_Layers[l].GetNeuronAt(n).GetParentOfSynapse().GetStartChildIndex();
			unsigned int endIndex = m_Layers[l].GetNeuronAt(n).GetParentOfSynapse().GetEndChildIndex();
			float weight = m_Layers[l].GetNeuronAt(n).GetParentOfSynapse().GetWeightGenerative();

			float sum = 0.0f;
			for (unsigned int i = startIndex; i <= endIndex; ++i)
				sum += (m_Layers[l + 1].GetNeuronAt(i).GetValue() * weight);

			m_Layers[l].FireInverseNeuronAt(n, sum);
		}
	}

	return GetLayerAt(1);
}

Layer ConvolutionalNeuralNetwork::Discriminate()
{
	return DiscriminateUntil(m_Layers.size());
}

Layer ConvolutionalNeuralNetwork::Generate(Layer input)
{
	return GenerateUntil(input, 1);
}

void ConvolutionalNeuralNetwork::LearnCurrentInput()
{
	Layer output = Discriminate();
	while (true)
	{
		ConvolutionalNeuralNetwork dupe = ConvolutionalNeuralNetwork(*this);

		if (dupe.Generate(output) == GetInput())
			break;

		for (unsigned int i = 1; i < m_Layers.size() - 1; ++i)
		{
			DiscriminateUntil(i);
			dupe.GenerateUntil(output, i);

			Layer change = GetLayerAt(i) - dupe.GetLayerAt(i);
			float result = 0.0f;
			
			for (unsigned int n = 1; n < change.GetNeurons().size(); ++n)
				result += pow(change.GetNeuronAt(n).GetValue(), 2);

			result = sqrt(result) * GetLearnRate();
			
			for (unsigned int j = 1; j < GetLayerAt(i - 1).GetNeurons().size(); ++j)
				GetLayerAt(i).IncrementParentWeightAt(j, result);
			for (unsigned int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
				GetLayerAt(i).IncrementParentWeightAt(j, -result);
		}
	}
}

float ConvolutionalNeuralNetwork::GetLearnRate()
{
	return m_LearnRate;
}

void ConvolutionalNeuralNetwork::SetLearnRate(float newValue)
{
	m_LearnRate = newValue;
}

void ConvolutionalNeuralNetwork::ReadFromFile(std::string path)
{
	std::ifstream file = std::ifstream(path);
	std::string contents = FindInBetween(std::string((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>()), "<[", "]>");
	
	std::string layer;
	unsigned int iterations = 0;
	m_Layers.clear();
	std::istringstream content(contents);
	while (std::getline(content, layer, '{'))
	{
		Layer newLayer;

		std::string neuron;
		std::istringstream current(layer);
		while (std::getline(current, neuron, ' '))
		{
			if (neuron != "")
			{
				std::string synapseData = FindInBetween(neuron, "<p", "p>");

				Synapse parentSynapse;

				std::string parent = FindInBetween(synapseData, "p(", ")p");
				int layerParent = std::stoi(StringUntil(parent, ","));
				int indexParent = std::stoi(StringBy(parent, ","));

				int childStartIndex = std::stoi(FindInBetween(synapseData, "cs(", ")sc"));
				int childEndIndex = std::stoi(FindInBetween(synapseData, "ce(", ")ec"));

				float weightD = std::stof(FindInBetween(synapseData, "d:", "g:"));
				float weightG = std::stof(FindInBetween(synapseData, "g:", ""));

				parentSynapse = Synapse(layerParent, indexParent, childStartIndex, childEndIndex);
				parentSynapse.SetWeightDiscriminate(weightD);
				parentSynapse.SetWeightGenerative(weightG);

				newLayer.AddNeuron(Neuron(parentSynapse));
			}
		}

		if (iterations < m_Layers.size())
			m_Layers[iterations] = newLayer;
		else
			AddLayer(newLayer);
		++iterations;
	}

	file.close();
}

void ConvolutionalNeuralNetwork::SaveToFile(std::string path)
{
	std::ofstream file = std::ofstream(path);
	file.clear();

	file << "<[";

	for (unsigned int i = 1; i < m_Layers.size(); ++i)
	{
		file << "{";

		for (unsigned int j = 1; j < m_Layers[i].GetNeurons().size() + 1; ++j)
		{
			file << " ";

			Synapse parent = m_Layers[i].GetNeuronAt(j).GetParentOfSynapse();

			file << "<p";
			file << "p(" << parent.GetParentLayer() << "," << parent.GetParentIndex() << ")p";
			file << "cs(" << parent.GetStartChildIndex() << ")sc";
			file << "ce(" << parent.GetEndChildIndex() << ")ec";
			file << "d:" << parent.GetWeightDiscriminate() << "g:" << parent.GetWeightGenerative();
			file << "p>";
		}

		file << "}";
	}

	file << "]>";
}

std::string ConvolutionalNeuralNetwork::StringUntil(std::string input, std::string key)
{
	std::string result;
	size_t index = input.rfind(key);
	if (index != std::string::npos)
	for (unsigned int i = 0; i < index; ++i)
		result += input[i];
	return result;
}

std::string ConvolutionalNeuralNetwork::StringBy(std::string input, std::string key)
{
	std::string result;
	size_t index = input.rfind(key);
	if (index != std::string::npos)
	for (unsigned int i = index + key.length(); i < input.length(); ++i)
		result += input[i];
	return result;
}

std::string ConvolutionalNeuralNetwork::FindInBetween(std::string input, std::string first, std::string second)
{
	std::string firstSeg = StringUntil(input, second);
	return StringBy(firstSeg, first);
}