#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

int main(int argc, char** argv)
{
	NeuralNet net = NeuralNet();
	net.add_layer(new ConvolutionLayer<1, 28, 28, 7, 2, 0, 0, 0>());
	net.add_layer(new ConvolutionLayer<1, 22, 22, 1, 2, 1, 28, 28>());
	net.add_layer(new MaxpoolLayer<1, 22, 22, 22, 1>());
	net.add_layer(new FeedForwardLayer<1, 22, 2000, 0>());
	net.add_layer(new FeedForwardLayer<1, 2000, 10, 22>());

	int t = clock();
	net.load_data("C:\\net.cnn");
	std::cout << "Elapsed time: " << clock() - t << std::endl;

	std::cout << "Type anything and press enter to exit" << std::endl;
	char c;
	std::cin >> c;
	return 0;
}