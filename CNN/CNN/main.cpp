#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

int main(int argc, char** argv)
{
	//Choose sample size to estime MSE
	NeuralNetAnalyzer::sample_size = 100;

	//setup the structure of the network
	NeuralNet net = NeuralNet();
	net.add_layer(new PerceptronFullConnectivityLayer<1, 1, 1, 1, 1, 1, false>());
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1, 2, false>());
	net.add_layer(new MaxpoolLayer<2, 1, 1, 1, 1>());
	net.add_layer(new PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, true>());
	net.add_layer(new SoftMaxLayer<1, 1, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	//choose parameters
	net.learning_rate = 0.05f;
	net.momentum_term = 0.005f;
	net.use_dropout = false;
	net.use_batch_learning = true;
	net.use_momentum = true;

	//get current process (weights and biases, not momentum)
	net.load_data("example.cnn");
	//this step is essential for learning
	net.setup_gradient();

	//don't forget to free later! Initialization lists 
	std::vector<IMatrix<float>*> input = { new Matrix2D<float, 1, 1>({ x }) };
	std::vector<IMatrix<float>*> labels = { new Matrix2D<float, 1, 1>({ y }) };

	//set up
	net.set_input(input);
	net.set_labels(labels);

	//alternating gibbs sampling 
	for (int i = 0; i < 100; ++i)
		net.pretrain(3);
	//save data
	net.save_data("example.cnn");

	for (int i = 0; i < 100; ++i)
	{
		//gradient checking
		std::vector<std::vector<IMatrix<float>*>> w_approx = NeuralNetAnalyzer::approximate_weight_gradient(net);
		std::vector<std::vector<IMatrix<float>*>> b_approx = NeuralNetAnalyzer::approximate_bias_gradient(net);

		//actual backprop (note that gradient is not applied (batch learning)
		net.train(3);
	
		//release 
		for (int l = 0; l < w_approx.size(); ++l)
			for (int d = 0; d < w_approx[l].size(); ++d)
				delete w_approx[l][d];
		for (int l = 0; l < w_approx.size(); ++l)
			for (int f_0 = 0; f_0 < w_approx[l].size(); ++f_0)
				delete b_approx[l][f_0];
	}
	if (net.use_batch_learning)
		net.apply_gradient();
	net.save_data("example.cnn");

	//test actual network
	for (int i = 0; i < 100; ++i)
		net.discriminate();

	delete input[0];
	delete labels[0];

	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}