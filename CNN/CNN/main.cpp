#include <iostream>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

int main(int argc, const char* args[])
{
	ConvolutionalNeuralNetwork neuralNetwork("secondCNN.cnn");

	float** input = new float*[2];
	for (int i = 0; i < 2; ++i)
		input[i] = new float[3];

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			input[i][j] = (rand() % 100) / 100;
	neuralNetwork.SetInput(input, 2, 3);

	Layer output = neuralNetwork.Discriminate();
	std::cout << "Output: " << output.GetNeuronAt(1).GetValue() << std::endl;
	std::cout << "Generative output: ";
	Layer generatedOutput = neuralNetwork.Generate(output);
	for (unsigned int i = 1; i < generatedOutput.GetNeurons().size(); ++i)
		std::cout << generatedOutput.GetNeuronAt(i).GetValue() << " ";
	std::cout << std::endl;

	char c;
	std::cin >> c;
	for (int i = 0; i < 2; ++i)
		delete[] input[i];
	delete[] input;
	return 0;
}