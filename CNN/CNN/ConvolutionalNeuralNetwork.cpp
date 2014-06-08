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
	std::map<SimpleNeuron, std::vector<Synapse>> childrenOf;
	for (unsigned int i = 0; i < neuronCountPerLayer.size() - 1; ++i)
	{
		Layer currentLayer;

		for (int j = 0; j < neuronCountPerLayer[i]; ++j)
		{
			std::vector<Synapse> parentOf;

			if (featureMapsPerLayer[i] == 1)
			{
				for (int n = 0; n < neuronCountPerLayer[i + 1]; ++n)
				{
					SimpleNeuron current = SimpleNeuron(i + 1, j + 1);
					SimpleNeuron destination = SimpleNeuron(i + 2, n + 1);

					Synapse currentParentSynapse = Synapse(current, current);
					Synapse currentChildSynapse = Synapse(destination, destination);

					currentChildSynapse.SetWeightDiscriminate(currentParentSynapse.GetWeightDiscriminate());
					currentChildSynapse.SetWeightGenerative(currentParentSynapse.GetWeightGenerative());

					parentOf.push_back(currentParentSynapse);

					if (childrenOf.find(destination) != childrenOf.end())
						childrenOf.at(destination).push_back(currentChildSynapse);
					else
						childrenOf.insert(std::pair<SimpleNeuron, std::vector<Synapse>>(destination,
						std::vector<Synapse>{ currentChildSynapse }));
				}
			}

			else
			{
				int featureMapsUp = featureMapsPerLayer[i + 1];
				int inFeatureMap = featureMapsPerLayer[i] / j;
				int connections = featureMapConnections[i][inFeatureMap];
				int startIndex = (neuronCountPerLayer[i + 1] / featureMapsUp) * featureMapStartIndex[i][inFeatureMap];
				int destinationIndex = startIndex + (neuronCountPerLayer[i + 1] / featureMapsUp) * connections;

				for (int n = startIndex; n < destinationIndex; ++n)
				{
					SimpleNeuron current = SimpleNeuron(i + 1, j + 1);
					SimpleNeuron destination = SimpleNeuron(i + 2, n + 1);

					Synapse currentParentSynapse = Synapse(current, current);
					Synapse currentChildSynapse = Synapse(destination, destination);

					currentChildSynapse.SetWeightDiscriminate(currentParentSynapse.GetWeightDiscriminate());
					currentChildSynapse.SetWeightGenerative(currentParentSynapse.GetWeightGenerative());

					parentOf.push_back(currentParentSynapse);

					if (childrenOf.find(destination) != childrenOf.end())
						childrenOf.at(destination).push_back(currentChildSynapse);
					else
						childrenOf.insert(std::pair<SimpleNeuron, std::vector<Synapse>>(destination,
						std::vector<Synapse>{ currentChildSynapse }));
				}
			}

			if (childrenOf.find(SimpleNeuron(i + 1, j + 1)) != childrenOf.end())
				currentLayer.AddNeuron(Neuron(parentOf, childrenOf.at(SimpleNeuron(i + 1, j + 1))));
			else
				currentLayer.AddNeuron(Neuron(parentOf, std::vector<Synapse>{}));
		}

		AddLayer(currentLayer);
	}

	Layer output;

	for (int i = 0; i < neuronCountPerLayer[neuronCountPerLayer.size() - 1]; ++i)
		output.AddNeuron(Neuron(std::vector<Synapse>(), childrenOf.at(SimpleNeuron(neuronCountPerLayer.size(), i + 1))));
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
	{
		for (unsigned int j = 0; j < input[i].size(); ++j)
		{
			std::vector<Synapse> parentOf;
			for (unsigned int n = 1; n < GetLayerAt(2).GetNeurons().size(); ++n)
				parentOf.push_back(Synapse(SimpleNeuron(1, 1 + i + (j * input.size())), SimpleNeuron(GetLayerAt(2).GetNeuronAt(n).GetLayer(),
				GetLayerAt(2).GetNeuronAt(n).GetIndex())));
			neurons.push_back(Neuron(parentOf, std::vector<Synapse>(0)));
		}
	}
	m_Layers[0] = Layer(neurons);
}

Layer ConvolutionalNeuralNetwork::GetOutput()
{
	return GetLayerAt(m_Layers.size());
}

Layer ConvolutionalNeuralNetwork::Discriminate()
{
	for (unsigned int i = 1; i < m_Layers.size() - 1; ++i)
		for (unsigned int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
			for (unsigned int s = 1; s < GetLayerAt(i).GetNeuronAt(j).GetChildOfSynapses().size(); ++s)
				GetLayerAt(i + 1).GetNeuronAt(GetLayerAt(i).GetNeuronAt(j).GetChildOfSynapseAt(s).GetChild().GetIndex())
				.SetValue(GetLayerAt(i).FireNeuronAt(j));
	return GetLayerAt(GetLayers().size());
}

Layer ConvolutionalNeuralNetwork::Generate(Layer input)
{
	for (int i = m_Layers.size(); i > 0; --i)
		for (int j = GetLayerAt(i).GetNeurons().size(); j > 0; --j)
			for (unsigned int s = 1; s < GetLayerAt(i).GetNeuronAt(j).GetParentOfSynapses().size(); ++s)
				m_Layers[i].GetNeuronAt(GetLayerAt(i).GetNeuronAt(j).GetParentOfSynapseAt(s).GetChild().GetIndex())
				.SetValue(GetLayerAt(i).FireInverseNeuronAt(j));
	return GetLayerAt(1);
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
			for (unsigned int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
				GetLayerAt(i).GetNeuronAt(j).FireSynapse();
			for (unsigned int n = dupe.GetLayerAt(i).GetNeurons().size(); n > i - 1; --n)
				for (int j = dupe.GetLayerAt(n).GetNeurons().size(); j > 0; --j)
					dupe.GetLayerAt(n).GetNeuronAt(j).FireInverseSynapse();

			Layer change = GetLayerAt(i) - dupe.GetLayerAt(i);
			float result = 0.0f;
			
			for (unsigned int n = 1; n < change.GetNeurons().size(); ++n)
				result += pow(change.GetNeuronAt(n).GetValue(), 2);

			result = sqrt(result) * GetLearnRate();
			
			for (unsigned int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
			{
				GetLayerAt(i).GetNeuronAt(j).IncrementParentWeights(result);
				GetLayerAt(i + 1).GetNeuronAt(j).DecrementChildWeights(result);
			}
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
			std::string parentSynapse = FindInBetween(neuron, "<p", "p>");
			std::string childSynapse = FindInBetween(neuron, "<c", "c>");

			std::vector<Synapse> parentSynapses;
			std::vector<Synapse> childSynapses;

			std::string synapse;
			std::istringstream parent(parentSynapse);
			while (std::getline(parent, synapse, '|'))
			{
				std::string synapseData = FindInBetween(synapse, "/", "\\");

				std::string parent = FindInBetween(synapseData, "p(", ")p");
				int layerParent = std::stoi(StringUntil(parent, ","));
				int indexParent = std::stoi(StringBy(parent, ","));

				std::string child = FindInBetween(synapseData, "c(", ")c");
				int layerChild = std::stoi(StringUntil(child, ","));
				int indexChild = std::stoi(StringBy(child, ","));

				float weightD = std::stof(FindInBetween(synapseData, "d:", "g:"));
				float weightG = std::stof(FindInBetween(synapseData, "g:", ""));

				Synapse newSynapse = Synapse(SimpleNeuron(layerParent, indexParent), SimpleNeuron(layerChild, indexChild));
				newSynapse.SetWeightDiscriminate(weightD);
				newSynapse.SetWeightGenerative(weightG);

				parentSynapses.push_back(newSynapse);
			}

			std::istringstream child(childSynapse);
			while (std::getline(child, synapse, '|'))
			{
				std::string synapseData = FindInBetween(synapse, "/", "\\");

				std::string parent = FindInBetween(synapseData, "p(", ")p");
				int layerParent = std::stoi(StringUntil(parent, ","));
				int indexParent = std::stoi(StringBy(parent, ","));

				std::string child = FindInBetween(synapseData, "c(", ")c");
				int layerChild = std::stoi(StringUntil(child, ","));
				int indexChild = std::stoi(StringBy(child, ","));

				float weightD = std::stof(FindInBetween(synapseData, "d:", "g:"));
				float weightG = std::stof(FindInBetween(synapseData, "g:", ""));

				Synapse newSynapse = Synapse(SimpleNeuron(layerParent, indexParent), SimpleNeuron(layerChild, indexChild));
				newSynapse.SetWeightDiscriminate(weightD);
				newSynapse.SetWeightGenerative(weightG);

				childSynapses.push_back(newSynapse);
			}

			newLayer.AddNeuron(Neuron(parentSynapses, childSynapses));
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

			file << "<p";
			for (unsigned int s = 1; s < m_Layers[i].GetNeuronAt(j).GetParentOfSynapses().size() + 1; ++s)
			{
				Synapse current = m_Layers[i].GetNeuronAt(j).GetParentOfSynapseAt(s);
				
				file << "/p(" << current.GetParent().GetLayer() << "," << current.GetParent().GetIndex() << ")p";
				file << "c(" << current.GetChild().GetLayer() << "," << current.GetChild().GetIndex() << ")c";
				file << "d:" << current.GetWeightDiscriminate() << "g:" << current.GetWeightGenerative() << "\\|";
			}
			file << "p>";

			file << "<c";
			for (unsigned int s = 1; s < m_Layers[i].GetNeuronAt(j).GetChildOfSynapses().size() + 1; ++s)
			{
				Synapse current = m_Layers[i].GetNeuronAt(j).GetChildOfSynapseAt(s);

				file << "/p(" << current.GetParent().GetLayer() << "," << current.GetParent().GetIndex() << ")p";
				file << "c(" << current.GetChild().GetLayer() << "," << current.GetChild().GetIndex() << ")c";
				file << "d:" << current.GetWeightDiscriminate() << "g:" << current.GetWeightGenerative() << "\\|";
			}
			file << "c>";
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