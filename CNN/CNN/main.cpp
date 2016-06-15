#include <conio.h>
#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

int main(int argc, char** argv)
{
	//Choose sample size to estimate error
	NeuralNetAnalyzer::sample_size = 100;

	//setup the structure of the network
	NeuralNet net = NeuralNet();
	net.add_layer(new InputLayer<1, 1, 1>());
	net.add_layer(new PerceptronFullConnectivityLayer<1, 1, 1, 1, 1, 1, CNN_FUNC_RELU>(10, -5));
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1, 2, CNN_FUNC_RELU>(3, -3));
	net.add_layer(new MaxpoolLayer<2, 1, 1, 1, 1>());
	net.add_layer(new PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, CNN_FUNC_RELU>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	//choose parameters, must be done before setup_gradient() or memory could be deallocated or uninitialized
	net.learning_rate = 0.01f;
	net.use_dropout = false;
	net.use_batch_learning = true;
	net.optimization_method = CNN_OPT_ADAM;
	net.loss_function = CNN_LOSS_QUADRATIC;

	//choose to disable a layer's biases, must be done before setup_gradient() or load so memory can be deallocated or not read
	net.layers[1]->use_biases = false;
	
	//this step is essential for learning
	net.setup_gradient();

	//get previous values (weights and biases only)
	net.load_data("example.cnn");

	//don't forget to free later! Initialization lists 
	std::vector<IMatrix<float>*> input = { new Matrix2D<float, 1, 1>({ 1 }) }; //basic input/output
	std::vector<IMatrix<float>*> labels = { new Matrix2D<float, 1, 1>({ 16 }) };

	//set up
	net.set_input(input);
	net.set_labels(labels);

	//alternating gibbs sampling 
	for (int i = 0; i < 100; ++i)
		net.pretrain(3);
	//save data
	net.save_data("example.cnn");

	float error = INFINITY;
	for (int batch = 0; error > .001f; ++batch)
	{
		for (int i = 0; i < 100; ++i)
		{
			//actual backprop (note that gradient is not applied (batch learning)
			NeuralNetAnalyzer::add_point(net.train());
		}
		if (net.use_batch_learning)
			net.apply_gradient();
		error = NeuralNetAnalyzer::mean_error();
		std::cout << "After " << batch << " batches, network has expected error of " << error << std::endl;
	}
	net.save_data("example.cnn");

	//test actual network
	input[0]->at(0, 0) = 2;
	net.set_input(input);
	net.discriminate();

	delete input[0];
	delete labels[0];

	std::cout << "\n\nPress any key to exit" << std::endl;
	_getche();
	return 0;
}