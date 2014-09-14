#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

int main(int argc, char** argv)
{
	NeuralNet net = NeuralNet();
	net.add_layer(new FeedForwardLayer(1, 5, 3, 0));
	net.add_layer(new FeedForwardLayer(1, 3, 2, 5));
	net.add_layer(new FeedForwardLayer(1, 2, 1, 3));
	Matrix2D<int> input[1];
	input[0] = Matrix2D<int>(5, 1, { 1, 2, 3, 4, 5 });
	net.set_input(input);
	int result = net.discriminate()->feature_maps[0].at(0, 0);
	std::cout << result << std::endl;

	input->at(3, 0) = 3;
	std::cout << input->at(3, 0) << std::endl;
	return 0;
}