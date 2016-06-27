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
	net.add_layer(new PerceptronFullConnectivityLayer<1, 1, 1, 1, 1, 1, CNN_FUNC_RELU>());
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1, 2, CNN_FUNC_RELU>(1, -1)); //custom weight range initialization (doesn't affect biases)
	net.add_layer(new MaxpoolLayer<2, 1, 1, 1, 1>());
	net.add_layer(new PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, CNN_FUNC_RELU>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	//choose parameters, must be done before setup_gradient() or memory could be deallocated or uninitialized
	net.learning_rate = 0.001f;
	net.use_dropout = false;
	net.use_l2_weight_decay = false; //because this flag is set, error will be much larger (adjust training benchmarks as necessary)
	net.include_bias_decay = true; //we want to discourage large biases
	net.weight_decay_factor = 0.001f;
	net.use_batch_learning = true;
	net.use_batch_normalization = true;
	net.keep_running_activation_statistics = false; //we are using minibatch statistics
	net.collect_data_while_training = false; //we are discriminate()-ing, not just train()-ing 
	net.optimization_method = CNN_OPT_ADAM;
	net.loss_function = CNN_LOSS_QUADRATIC;

	//choose to disable a layer's biases, must be done before setup_gradient() or load so memory can be deallocated or not read
	net.layers[1]->use_biases = false;
	
	//this step is essential for learning
	net.setup_gradient();

	//don't forget to free later! Initialization lists 
	std::vector<IMatrix<float>*> input = { new Matrix2D<float, 1, 1>({ 1 }) }; //basic input/output
	std::vector<IMatrix<float>*> labels = { new Matrix2D<float, 1, 1>({ 1 }) };

	//set up
	net.set_input(input);
	net.set_labels(labels);

	//get gradient error
	net.train();
	std::pair<float, float> errors = NeuralNetAnalyzer::mean_gradient_error(net, net.weights_gradient, net.biases_gradient);
	std::cout << "Approximate gradient errors: " << errors.first << ',' << errors.second << std::endl; //errors will be enormous if using batch normalization since it modifies the data

	//save data
	net.save_data("example.cnn");

	float error = INFINITY;
	for (int batch = 0; error > 1; ++batch)
	{
		for (int i = 0; i < 100; ++i)
		{
			//since we are using minibatch normalization and NOT keeping a running total of the statistics,
			//then we must run each sample from the minibatch through the network to collect the data
			input[0]->at(0, 0) = rand() % 20;
			labels[0]->at(0, 0) = input[0]->at(0, 0);
			net.set_input(input);
			net.set_labels(labels);

			net.discriminate(); //no special function calls needed to get data for minibatch statistics, discriminate() and flags set above will do it all
		}
		for (int i = 0; i < 100; ++i)
		{
			//actual backprop, note that gradient is not applied (batch learning)
			input[0]->at(0, 0) = rand() % 20;
			labels[0]->at(0, 0) = input[0]->at(0, 0);
			net.set_input(input);
			net.set_labels(labels);

			NeuralNetAnalyzer::add_point(net.train()); //if we were using all learning examples for batch normalization, then we would want to set the 
			                                           //collect_data_while_training flag to true, since we never call discriminate()
		}
		if (net.use_batch_learning)
			net.apply_gradient();
		error = NeuralNetAnalyzer::mean_error(); //will factor in L2 weight error
		std::cout << "After " << batch << " batches, network has expected error of " << error << std::endl;
	}
	net.save_data("example.cnn");

	//test actual network
	input[0]->at(0, 0) = 2;
	net.set_input(input);
	net.discriminate();
	std::cout << "Net value with input of 2: " << net.layers[net.layers.size() - 1]->feature_maps[0]->at(0, 0) << std::endl;

	//get previous values (weights and biases only) if desired
	net.load_data("example.cnn");

	delete input[0];
	delete labels[0];

	std::cout << "\n\nPress any key to exit" << std::endl;
	_getche();
	return 0;
}