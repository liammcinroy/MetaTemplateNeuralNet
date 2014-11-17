#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	NeuralNet net = NeuralNet();
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1>());
	net.add_layer(new MaxpoolLayer<1, 1, 1, 1, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	net.learning_rate = 0.05f;
	net.use_dropout = false;
	net.binary_net = false;

	net.load_data("C://example.cnn");

	std::vector<Matrix<float>*> input = { new Matrix2D<float, 1, 1>({ 1 }) };
	std::vector<Matrix<float>*> labels = { new Matrix2D<float, 1, 1>({ 3 }) };

	net.set_input(input);
	net.set_labels(labels);

	for (int i = 0; i < 100; ++i)
		net.train(3);
	
	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}