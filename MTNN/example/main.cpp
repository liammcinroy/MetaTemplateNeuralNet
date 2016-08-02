#include <conio.h>
#include <iostream>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

//setup the structure of the network
typedef NeuralNet<CNN_LOSS_SQUAREERROR, CNN_OPT_BACKPROP, false, true, false, false, true, false, true, false, false,
	InputLayer<1, 1, 1, 1>, //the indexes allow the classes to be static
	PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, true>,
	ConvolutionLayer<3, 1, 1, 1, 1, 1, 2, CNN_FUNC_RELU, false, true>, //can disable biases and padding
	MaxpoolLayer<4, 2, 1, 1, 1, 1>,
	PerceptronFullConnectivityLayer<5, 2, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, true>,
	OutputLayer<6, 1, 1, 1>> Net;

int main(int argc, char** argv)
{
	//Have to define input/output filename before template because templates don't take string literals and needs linking
	auto path = CSTRING("example.cnn");

	//Choose sample size to estimate error
	NeuralNetAnalyzer<Net>::sample_size = 100;

	Net::learning_rate = .0001f;

	//when calling setup() memory could be deallocated or uninitialized
	//this step is maybe? essential for learning
	Net::setup();

	//basic input/output
	FeatureMaps<1, 1, 1> input{};
	FeatureMaps<1, 1, 1> labels{};

	//set up
	Net::set_input(input);
	Net::set_labels(labels);

	//get gradient error
	Net::train();
	std::pair<float, float> errors = NeuralNetAnalyzer<Net>::mean_gradient_error();
	std::cout << "Approximate gradient errors: " << errors.first << ',' << errors.second << std::endl; //errors will be enormous if using batch normalization since it modifies the data

	//save data this will give the type of path which has a static variable with the string of path
	Net::save_data<decltype(path)>();

	float error = INFINITY;
	for (int batch = 1; error > 1; ++batch)
	{
		for (int i = 0; i < 100; ++i)
		{
			//since we are using minibatch normalization and NOT keeping a running total of the statistics,
			//then we must run each sample from the minibatch through the network to collect the data
			input[0].at(0, 0) = rand() % 20;
			labels[0].at(0, 0) = input[0].at(0, 0);
			Net::set_input(input);
			Net::set_labels(labels);

			Net::discriminate(); //no special function calls needed to get data for minibatch statistics, discriminate() and flags set above will do it all
		}
		for (int i = 0; i < 100; ++i)
		{
			//actual backprop, note that gradient is not applied (batch learning)
			input[0].at(0, 0) = rand() % 20;
			labels[0].at(0, 0) = input[0].at(0, 0);
			Net::set_input(input);
			Net::set_labels(labels);
			float w = Net::template get_layer<1>::weights[0].at(0, 0);
			float g = Net::template get_layer<1>::weights_gradient[0].at(0, 0);

			NeuralNetAnalyzer<Net>::add_point(Net::train()); //if we were using all learning examples for batch normalization, then we would want to set the 
															 //collect_data_while_training flag to true, since we never call discriminate()
		}
		Net::apply_gradient();//since we are using batch learning
		error = NeuralNetAnalyzer<Net>::mean_error();
		std::cout << "After " << batch << " batches, network has expected error of " << error << std::endl;
	}
	Net::save_data<decltype(path)>();

	//test actual network
	input[0].at(0, 0) = 2;
	Net::set_input(input);
	Net::discriminate();
	std::cout << "Net value with input of 2: " << (float)(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;

	//get previous values (weights and biases only) if desired
	Net::load_data<decltype(path)>();

	std::cout << "\n\nPress any key to exit" << std::endl;
	_getche();
	return 0;
}