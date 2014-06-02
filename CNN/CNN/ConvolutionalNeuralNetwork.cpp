#include "ConvolutionalNeuralNetwork.h"

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork()
{
}

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::string path)
{
	ReadFromFile(path);
}

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork()
{
}

void ConvolutionalNeuralNetwork::AddLayer(Layer newLayer)
{
	m_Layers.push_back(newLayer);
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

void ConvolutionalNeuralNetwork::SetInput(float** input, int width, int height)
{
	std::vector<Neuron> neurons;
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			std::vector<Synapse> parentOf;
			for (int n = 1; n < GetLayerAt(2).GetNeurons().size(); ++n)
				parentOf.push_back(Synapse(SimpleNeuron(1, 1 + i + (j * width)), SimpleNeuron(GetLayerAt(2).GetNeuronAt(n).GetLayer(),
				GetLayerAt(1).GetNeuronAt(n).GetIndex())));
			neurons.push_back(Neuron(parentOf, std::vector<Synapse>(0)));
		}
	}
	m_Layers[0] = Layer(neurons);
}

Layer ConvolutionalNeuralNetwork::GetOutput()
{
	return GetLayerAt(m_Layers.size());
}

void ConvolutionalNeuralNetwork::SetOutput(float** newOutput, int width, int height)
{
	std::vector<Neuron> neurons;
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			std::vector<Synapse> parentOf;
			for (int n = 1; n < GetLayerAt(2).GetNeurons().size(); ++n)
				parentOf.push_back(Synapse(SimpleNeuron(1, 1 + i + (j * width)), SimpleNeuron(GetLayerAt(2).GetNeuronAt(n).GetLayer(),
				GetLayerAt(1).GetNeuronAt(n).GetIndex())));
			neurons.push_back(Neuron(parentOf, std::vector<Synapse>(0)));
		}
	}
	m_Layers[m_Layers.size() - 1] = Layer(neurons);
}

Layer ConvolutionalNeuralNetwork::Discriminate()
{
	for (int i = 1; i < m_Layers.size(); ++i)
		for (int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
			GetLayerAt(i).GetNeuronAt(j).FireSynapse();
	return GetLayerAt(GetLayers().size());
}

Layer ConvolutionalNeuralNetwork::Generate(Layer input)
{
	for (int i = m_Layers.size(); i > 0; --i)
		for (int j = GetLayerAt(i).GetNeurons().size(); j > 0; --j)
			GetLayerAt(i).GetNeuronAt(j).FireInverseSynapse();
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

		for (int i = 1; i < m_Layers.size() - 1; ++i)
		{
			for (int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
				GetLayerAt(i).GetNeuronAt(j).FireSynapse();
			for (int n = dupe.GetLayerAt(i).GetNeurons().size(); n > i - 1; --n)
				for (int j = dupe.GetLayerAt(n).GetNeurons().size(); j > 0; --j)
					dupe.GetLayerAt(n).GetNeuronAt(j).FireInverseSynapse();

			Layer change = GetLayerAt(i) - dupe.GetLayerAt(i);
			float result = 0.0f;
			
			for (int n = 1; n < change.GetNeurons().size(); ++n)
				result += pow(change.GetNeuronAt(n).GetOutput(), 2);

			result = sqrt(result) * GetLearnRate();
			
			for (int j = 1; j < GetLayerAt(i).GetNeurons().size(); ++j)
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
	std::string contents = findInBetween(std::string((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>()), "<[", "]>");
	
	std::string layer;
	int iterations = 1;
	while (std::getline(std::istringstream(contents), layer, '{'))
	{
		Layer newLayer;

		std::string neuron;
		while (std::getline(std::istringstream(layer), neuron, ' '))
		{
			std::string parentSynapse = findInBetween(neuron, "<p", "p>");
			std::string childSynapse = findInBetween(neuron, "<c", "c>");

			std::vector<Synapse> parentSynapses;
			std::vector<Synapse> childSynapses;

			std::string synapse;
			while (std::getline(std::istringstream(parentSynapse), synapse, '|'))
			{
				std::string synapseData = findInBetween(synapse, "/", "\\");

				std::string parent = findInBetween(synapseData, "p(", ")p");
				int layerParent = stoi(stringUntil(parent, parent.find(',')));
				int indexParent = stoi(stringBy(parent, parent.rfind(',')));

				std::string child = findInBetween(synapseData, "c(", ")c");
				int layerChild = stoi(stringUntil(child, child.find(',')));
				int indexChild = stoi(stringBy(child, child.rfind(',')));

				float weightD = stof(findInBetween(synapseData, "d:", "g:"));
				float weightG = stof(findInBetween(synapseData, "g:", ""));

				Synapse newSynapse = Synapse(SimpleNeuron(layerParent, indexParent), SimpleNeuron(layerChild, indexChild));
				newSynapse.SetWeightDiscriminate(weightD);
				newSynapse.SetWeightGenerative(weightG);

				parentSynapses.push_back(newSynapse);
			}

			while (std::getline(std::istringstream(childSynapse), synapse, '|'))
			{
				std::string synapseData = findInBetween(synapse, "/", "\\");

				std::string parent = findInBetween(synapseData, "p(", ")p");
				int layerParent = stoi(stringUntil(parent, parent.find(',')));
				int indexParent = stoi(stringBy(parent, parent.rfind(',')));

				std::string child = findInBetween(synapseData, "c(", ")c");
				int layerChild = stoi(stringUntil(child, child.find(',')));
				int indexChild = stoi(stringBy(child, child.rfind(',')));

				float weightD = stof(findInBetween(synapseData, "d:", "g:"));
				float weightG = stof(findInBetween(synapseData, "g:", ""));

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
			m_Layers.push_back(newLayer);
		++iterations;
	}

	file.close();
}

void ConvolutionalNeuralNetwork::SaveToFile(std::string path)
{
	std::ofstream file = std::ofstream(path);
	file.clear();

	file << "<[";

	for (int i = 1; i < m_Layers.size(); ++i)
	{
		file << "{";

		for (int j = 1; j < m_Layers[i].GetNeurons().size() + 1; ++j)
		{
			file << " ";

			file << "<p";
			for (int s = 1; s < m_Layers[i].GetNeuronAt(j).GetParentOfSynapses().size() + 1; ++s)
			{
				Synapse current = m_Layers[i].GetNeuronAt(j).GetParentOfSynapseAt(s);
				
				file << "/p(" << current.GetParent().GetLayer() << "," << current.GetParent().GetIndex() << ")p";
				file << "c(" << current.GetChild().GetLayer() << "," << current.GetChild().GetIndex() << ")c";
				file << "d:" << current.GetWeightDiscriminate() << "g:" << current.GetWeightGenerative() << "\\|";
			}
			file << "p>";

			file << "<c";
			for (int s = 1; s < m_Layers[i].GetNeuronAt(j).GetChildOfSynapses().size() + 1; ++s)
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

std::string stringUntil(std::string input, size_t index)
{
	std::string result;
	if (index != std::string::npos)
		for (int i = 0; i < index; --i)
			result += input[i];
	return result;
}

std::string stringBy(std::string input, size_t index)
{
	std::string result;
	if (index != std::string::npos)
		for (int i = index; i < input.length; ++i)
			result += input[i];
	return result;
}

std::string findInBetween(std::string input, std::string first, std::string second)
{
	std::string firstSeg = stringUntil(input, input.find(second));
	return stringBy(firstSeg, firstSeg.rfind(first));
}