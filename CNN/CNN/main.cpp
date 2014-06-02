#include <iostream>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Synapse.h"
#include "SimpleNeuron.h"

std::string StringUntil(std::string input, std::string key)
{
	std::string result;
	size_t index = input.rfind(key);
	if (index != std::string::npos)
		for (unsigned int i = 0; i < index; ++i)
			result += input[i];
	return result;
}

std::string StringBy(std::string input, std::string key)
{
	std::string result;
	size_t index = input.rfind(key);
	if (index != std::string::npos)
		for (unsigned int i = index + key.length(); i < input.length(); ++i)
			result += input[i];
	return result;
}

std::string FindInBetween(std::string input, std::string first, std::string second)
{
	std::string firstSeg = StringUntil(input, second);
	return StringBy(firstSeg, first);
}

int main(int argc, const char* args[])
{
	ConvolutionalNeuralNetwork neuralNetwork(path);
	char c;
	std::cin >> c;
	return 0;
}