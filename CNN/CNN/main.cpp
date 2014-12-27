#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	NeuralNet net = NeuralNet();
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 2>());
	net.add_layer(new ConvolutionLayer<2, 1, 1, 1, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	net.learning_rate = 0.05f;
	net.use_dropout = false;
	net.binary_net = false;

	std::vector<Matrix<float>*> input = { new Matrix2D<float, 1, 1>({ 1 }) };
	std::vector<Matrix<float>*> labels = { new Matrix2D<float, 1, 1>({ 3 }) };

	net.set_input(input);
	net.set_labels(labels);

	for (int i = 0; i < 50; ++i)
	{
		net.pretrain(5000);
		float randN = (rand() % 10) + 1;
		input = { new Matrix2D<float, 1, 1>({ randN }) };
		labels = { new Matrix2D<float, 1, 1>({ 3 * randN }) };

		net.set_input(input);
		net.set_labels(labels);

		std::cout << net.discriminate()->feature_maps[0]->at(0, 0) / randN << std::endl;
	}

	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}