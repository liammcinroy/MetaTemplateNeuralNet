#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

int main(int argc, char** argv)
{
	NeuralNet net = NeuralNet();
	net.add_layer(new PerceptronLayer<1, 1, 1, 1, 1, 1>());
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1, 2>());
	net.add_layer(new MaxpoolLayer<2, 1, 1, 1, 1>());
	net.add_layer(new PerceptronLayer<2, 1, 1, 1, 1, 1>());
	net.add_layer(new SoftMaxLayer<1, 1, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	net.learning_rate = 0.05f;
	net.use_dropout = false;
	net.binary_net = false;

	net.load_data("example.cnn");

	std::vector<IMatrix<float>*> input = { new Matrix2D<float, 1, 1>({ x }) };
	std::vector<IMatrix<float>*> labels = { new Matrix2D<float, 1, 1>({ y }) };

	net.set_input(input);
	net.set_labels(labels);

	for (int i = 0; i < 100; ++i)
		net.pretrain(3);
	net.save_data("example.cnn");

	for (int i = 0; i < 100; ++i)
		net.train(3);
	net.save_data("example.cnn");

	for (int i = 0; i < 100; ++i)
		net.discriminate();

	delete labels[0];

	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}