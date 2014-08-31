#include <iostream>
#include <string>
#include <time.h>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Matrix.h"

int main(int argc, const char* args[])
{
	std::cout << "Creating Network..." << std::endl;
	int time = clock();
	convolutional_neural_network network(false);
	matrix<float> kernel;
	kernel = {
		{ 1, 1, 1 },
		{ 0, 0, 0 },
		{ -1, -1, -1 } };
	//network.push_layer(layer(1, 5, 5, CNN_CONVOLUTION, 3, kernel));
	//network.push_layer(layer(1, 1, 7, CNN_FEED_FORWARD, 5, 7, 1, 1));
	//network.push_layer(layer(1, 1, 5, CNN_FEED_FORWARD, 3, 5, 1, 1));
	//network.push_layer(layer(1, 1, 3, CNN_OUTPUT, 0));
	network.push_layer(layer(1, 1, 3, CNN_FEED_FORWARD, 2, 3, 1, 1));
	network.push_layer(layer(1, 1, 2, CNN_OUTPUT, 0, 0, 0, 0));
	time = clock() - time;
	std::cout << "Time to build: " << time << "\n" << std::endl;

	std::cout << "Discriminating input..." << std::endl;
	time = clock();
	matrix<float> input(1, 3, 1);
	input.set(0, 0, 0, 1);
	input.set(1, 0, 0, 2);
	input.set(2, 0, 0, 3);

	//input = { 1, 2, 3, 4, 5, 6, 7 };
	/*input = {
		{ 5, 5, 5, 5, 5 },
		{ 4, 4, 4, 4, 4 },
		{ 3, 3, 3, 3, 3 },
		{ 2, 2, 2, 2, 2 },
		{ 1, 1, 1, 1, 1 } };*/
	network.input = input;
	matrix<float> discriminated = network.discriminate().at(0);
	time = clock() - time;

	std::cout << "Output:\n" << discriminated.to_string() << "\n";
	std::cout << "Time to discriminate: " << time << "\n" << std::endl;

	std::cout << "Generating last labels..." << std::endl;
	time = clock();
	matrix<float> generated = network.generate(discriminated).at(0);
	time = clock() - time;

	std::cout << "Output:\n" << generated.to_string() << "\n";
	std::cout << "Time to generate: " << time << "\n" << std::endl;

	char c;
	std::cin >> c;
	return 0;
}