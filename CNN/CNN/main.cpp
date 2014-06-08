#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

int main(int argc, const char* args[])
{
	std::vector<int> numNeurons = { 500, 500, 2000, 10 };
	std::vector<int> numMaps = { 1, 1, 1, 1 };

	ConvolutionalNeuralNetwork neuralNetwork(numNeurons, numMaps, numNeurons, std::vector<std::vector<int>>(), std::vector<std::vector<int>>());

	std::cout << "Layers: " << neuralNetwork.GetLayers().size() << ", Neurons on output: " << neuralNetwork.GetOutput().GetNeurons().size() << std::endl;

	neuralNetwork.SaveToFile("test2.cnn");

	std::cout << "Saved" << std::endl;

	std::vector<std::vector<float>> input;
	for (int i = 0; i < 2; ++i)
		input.push_back(std::vector<float>{});

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			input[i].push_back(rand() % 100);
	neuralNetwork.SetInput(input);

	Layer output = neuralNetwork.Discriminate();
	std::cout << "Output: " << output.GetNeuronAt(1).GetValue() << std::endl;
	std::cout << "Generative output: ";
	Layer generatedOutput = neuralNetwork.Generate(output);
	for (unsigned int i = 1; i < generatedOutput.GetNeurons().size(); ++i)
		std::cout << generatedOutput.GetNeuronAt(i).GetValue() << " ";
	std::cout << std::endl;

	char c;
	std::cin >> c;
	return 0;
}