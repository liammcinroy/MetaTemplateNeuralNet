#include <conio.h>
#include <iostream>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

#define BATCH_SIZE 100
#define BATCH_FUNCTIONS true //use to compare using batch functions vs automatic batch learning
							//Note that using the automatic batch learnin does not make sense with batch norm and is ill defined

//setup the structure of the network
typedef NeuralNet<
	InputLayer<1, 1, 1, 1>, //the indexes allow the classes to be static
	BatchNormalizationLayer<1, 1, 1, 1, CNN_FUNC_TANH>,
	PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, false>,
	ConvolutionLayer<3, 1, 1, 1, 1, 1, 2, CNN_FUNC_LINEAR, false, false>, //can disable biases and padding
	BatchNormalizationLayer<3, 2, 1, 1, CNN_FUNC_LOGISTIC>,
	MaxpoolLayer<4, 2, 1, 1, 1, 1>,
	PerceptronFullConnectivityLayer<5, 2, 1, 1, 1, 1, 1,  CNN_FUNC_LINEAR, true>,
	PerceptronFullConnectivityLayer<6, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, true>, //Because of different indexes, then this and layer 1 won't share data
	OutputLayer<7, 1, 1, 1>> Net;

FeatureMap<1, 1, 1> PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, false>::weights = { .1f };//custom weight initialization

int main(int argc, char** argv)
{
	//Have to define input/output filename before template because templates don't take string literals and needs linking
	auto path = CSTRING("example.cnn");

	Net::loss_function = CNN_LOSS_SQUAREERROR;
	Net::optimization_method = CNN_OPT_BACKPROP;
	Net::learning_rate = .0001f;
	Net::use_batch_learning = true;
	Net::weight_decay_factor = .001f;
	Net::use_l2_weight_decay = false;
	Net::include_bias_decay = false;

	//Choose sample size to estimate error
	if (BATCH_FUNCTIONS)
		NeuralNetAnalyzer<Net>::sample_size = 1;
	else
		NeuralNetAnalyzer<Net>::sample_size = BATCH_SIZE;

	//when calling setup() memory could be deallocated or uninitialized
	//this step is maybe? essential for learning
	Net::setup();

	//basic input/output
	auto input = FeatureMapVector<1, 1, 1>(BATCH_SIZE);
	auto labels = FeatureMapVector<1, 1, 1>(BATCH_SIZE);

	//get gradient error
	Net::train();
	std::pair<float, float> errors = NeuralNetAnalyzer<Net>::mean_gradient_error();
	std::cout << "Approximate gradient errors: " << errors.first << ',' << errors.second << std::endl; //errors will be enormous if using batch normalization since it modifies the data


	//save data this will give the type of path which has a static variable with the string of path
	Net::save_data<decltype(path)>();

	float error = INFINITY;
	for (int batch = 1; error > .1; ++batch)
	{
		for (int i = 0; i < BATCH_SIZE; ++i)
		{
			//since we are using minibatch normalization and NOT keeping a running total of the statistics,
			//then we must run each sample from the minibatch through the network to collect the data
			input[i][0].at(0, 0) = (i + 1) % 10;
			labels[i][0].at(0, 0) = (i + 1) % 10;


			if (!BATCH_FUNCTIONS) //if using automated, just train on every input
			{
				Net::set_input(input[i]);
				Net::set_labels(labels[i]);
				NeuralNetAnalyzer<Net>::add_point(Net::train());
			}
		}
		
		if (BATCH_FUNCTIONS)
			NeuralNetAnalyzer<Net>::add_point(Net::train_batch(input, labels)); //if using batch, pass batch inputs

		else
			Net::apply_gradient(); //apply gradient if using automated

		if (Net::learning_rate > .0001f)
			Net::learning_rate *= .9;
		error = NeuralNetAnalyzer<Net>::mean_error();
		std::cout << "After " << batch << " batches, network has expected error of " << error << std::endl;
	}
	Net::save_data<decltype(path)>();

	//test actual network
	Net::set_input(FeatureMap<1, 1, 1>{1});
	Net::discriminate();
	std::cout << "Net value with input of 1: " << (float)(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;

	//get previous values (weights and biases only) if desired
	Net::load_data<decltype(path)>();

	Net::discriminate();
	std::cout << "Net value with input of 1 (after load): " << (float)(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;


	std::cout << "\n\nPress any key to exit" << std::endl;
	_getche();
	return 0;
}