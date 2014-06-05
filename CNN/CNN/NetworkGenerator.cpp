#include <map>
#include <vector>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

ConvolutionalNeuralNetwork CreateNetwork(int neuronCountPerLayer[], int featureMapsPerLayer[])
{
	ConvolutionalNeuralNetwork network;
	std::map<SimpleNeuron, std::vector<Synapse>> childrenOf;

	for (int i = 0; i < (sizeof(neuronCountPerLayer) / sizeof(*neuronCountPerLayer)); ++i)
	{
		Layer currentLayer;
		
		for (int j = 0; j < neuronCountPerLayer[i]; ++j)
		{
			std::vector<Synapse> parentOf;

			if (featureMapsPerLayer[i] == 1)
			{
				for (int n = 0; n < neuronCountPerLayer[i + 1]; ++n)
				{
					Synapse currentParentSynapse = Synapse(SimpleNeuron(i + 1, j + 1), SimpleNeuron(i + 2, n + 1));
					Synapse currentChildSynapse = Synapse(SimpleNeuron(i + 2, n + 1), SimpleNeuron(i + 1, j + 1));

					currentChildSynapse.SetWeightDiscriminate(currentParentSynapse.GetWeightDiscriminate());
					currentChildSynapse.SetWeightGenerative(currentParentSynapse.GetWeightGenerative());

					parentOf.push_back(currentParentSynapse);

					if (childrenOf.find(SimpleNeuron(i + 2, n + 1)) != childrenOf.end())
						childrenOf.at(SimpleNeuron(i + 2, n + 1)).push_back(currentChildSynapse);
					else
						childrenOf.insert(std::pair<SimpleNeuron, std::vector<Synapse>>(SimpleNeuron(i + 2, n + 1),
						std::vector<Synapse>{ currentChildSynapse }));
				}
			}

			currentLayer.AddNeuron(Neuron(parentOf, childrenOf.at(SimpleNeuron(i + 1, j + 1))));
		}

		network.AddLayer(currentLayer);
	}

	return network;
}