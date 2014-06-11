#include <time.h>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"

int main(int argc, const char* args[])
{
	
	//Example of how to initially create (only once)
	std::vector<int> numNeurons = { 500, 500, 2000, 10 };
	std::vector<int> numMaps = { 1, 1, 1, 1 };

	int t = clock();
	ConvolutionalNeuralNetwork neuralNetwork(numNeurons, numMaps, numNeurons, std::vector<std::vector<int>>(), std::vector<std::vector<int>>());
	//ConvolutionalNeuralNetwork neuralNetwork("handwriting.cnn");

	t = clock() - t;
	std::cout << "Clocks to build: " << t << std::endl;

	std::cout << "Layers: " << neuralNetwork.GetLayers().size() << ", Neurons on output: " << neuralNetwork.GetOutput().GetNeurons().size() << std::endl;

	t = clock();
	neuralNetwork.SaveToFile("test2.cnn");
	//Don't forget to save!
    t = clock() - t;
	std::cout << "Saved in " << t << " clocks." << std::endl;

	std::vector<std::vector<float>> input;
	for (int i = 0; i < 2; ++i)
		input.push_back(std::vector<float>{});

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			input[i].push_back(rand() % 100);
	neuralNetwork.SetInput(input);

	t = clock();
	Layer output = neuralNetwork.Discriminate();
	t = clock() - t;
	std::cout << "Clocks to discriminate: " << t << std::endl;
	std::cout << "Output: " << output.GetNeuronAt(1).GetValue() << std::endl;

	t = clock();
	Layer generatedOutput = neuralNetwork.Generate(output);
	t = clock() - t;
	std::cout << "Clocks to generate: " << t << std::endl;

	std::cout << "Generative output: ";
	for (unsigned int i = 1; i < generatedOutput.GetNeurons().size(); ++i)
		std::cout << generatedOutput.GetNeuronAt(i).GetValue() << " ";
	std::cout << std::endl;

	char c;
	std::cin >> c;
	return 0;
}