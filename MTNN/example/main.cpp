#include <conio.h>
#include <iostream>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

#define TRAINING true

#define BATCH_SIZE 100
#define BATCH_FUNCTIONS true //use to compare using batch functions vs automatic batch learning
//Note that using the automatic batch learnin does not make sense with batch norm and is ill defined

#define INPUT_TRANSFORM(x) x //((x - 4.5f) / 2.872f) //test bn
#define OUTPUT_TRANSFORM(x) x
#define OUTPUT_INV_TRANSFORM(x) x

//setup the structure of the network
typedef NeuralNet<
	InputLayer<1, 1, 1, 1>, //the indexes allow the classes to be static
	BatchNormalizationLayer<1, 1, 1, 1, CNN_FUNC_LINEAR>, //could use to normalize inputs
	PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, false>, //can disable biases
	ConvolutionLayer<3, 1, 1, 1, 1, 1, 2, CNN_FUNC_LINEAR, true, false>, //can disable padding
	BatchNormalizationLayer<3, 2, 1, 1, CNN_FUNC_TANH>, //if want to use tanh for conv layer with bn, use linear on conv then logistic for bn
	MaxpoolLayer<4, 2, 1, 1, 1, 1>,
	PerceptronFullConnectivityLayer<5, 2, 1, 1, 1, 1, 1, CNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<6, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, true>, //Because of different indexes, then this and layer 1 won't share data
	OutputLayer<7, 1, 1, 1>,
	OutputLayer<8, 1, 1, 1>> Net;

//typedef NeuralNet<
//	InputLayer<1, 1, 1, 1>,
//	BatchNormalizationLayer<1, 1, 1, 1, CNN_FUNC_LINEAR>, //could use to normalize inputs
//	PerceptronFullConnectivityLayer<2, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, false>, //can disable biases
//	BatchNormalizationLayer<2, 1, 1, 1, CNN_FUNC_LINEAR>,
//	PerceptronFullConnectivityLayer<6, 1, 1, 1, 1, 1, 1, CNN_FUNC_LINEAR, true>, //Because of different indexes, then this and layer 1 won't share data
//	OutputLayer<7, 1, 1, 1>> Net;

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
	auto inputs = FeatureMapVector<1, 1, 1>(BATCH_SIZE);
	auto labels = FeatureMapVector<1, 1, 1>(BATCH_SIZE);

	//get gradient error, won't mean much because online training will kill any BN network
	Net::train();
	std::pair<float, float> errors = NeuralNetAnalyzer<Net>::mean_gradient_error();
	std::cout << "Approximate gradient errors: " << errors.first << ',' << errors.second << std::endl; //errors will be enormous if using batch normalization since it modifies the data

	if (!TRAINING)
		Net::load_data<decltype(path)>();

	float error = INFINITY;
	for (int batch = 1; error > .001f && TRAINING; ++batch)
	{
		for (int i = 0; i < BATCH_SIZE; ++i)
		{
			//since we are using minibatch normalization and NOT keeping a running total of the statistics,
			//then we must run each sample from the minibatch through the network to collect the data
			inputs[i][0].at(0, 0) = INPUT_TRANSFORM((i + 1) % 10);
			labels[i][0].at(0, 0) = OUTPUT_TRANSFORM((i + 1) % 10);


			if (!BATCH_FUNCTIONS) //if using automated, just train on every input
			{
				Net::set_input(inputs[i]);
				Net::set_labels(labels[i]);
				NeuralNetAnalyzer<Net>::add_point(Net::train());
			}
		}

		if (BATCH_FUNCTIONS)
			NeuralNetAnalyzer<Net>::add_point(Net::train_batch(inputs, labels)); //if using bn/batch functions, pass batch inputs

		else
			Net::apply_gradient(); //apply gradient if using automated

		if (Net::learning_rate > .0001f)
			Net::learning_rate *= .9;
		error = NeuralNetAnalyzer<Net>::mean_error();
		std::cout << "After " << batch << " batches, network has expected error of " << error << std::endl;

		std::cout << "Net value with input (minibatch statistics) of 1: " << OUTPUT_INV_TRANSFORM(Net::template get_batch_activations<Net::last_layer_index>()[0][0].at(0, 0)) << std::endl;

		//test actual network (difference in values is due to changed weights, etc.)
		Net::set_input(FeatureMap<1, 1, 1>{ INPUT_TRANSFORM(1.0f) });

		Net::discriminate();
		std::cout << "Net value with input (using population averages) of 1: " << OUTPUT_INV_TRANSFORM(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;
	}
	Net::calculate_population_statistics(inputs); //Find the entire training sets batch statistics after training is done
	Net::save_data<decltype(path)>(); //save to path

									  //test actual network
	Net::set_input(FeatureMap<1, 1, 1>{ INPUT_TRANSFORM(1.0f) });
	Net::discriminate();
	std::cout << "Net value with input of 1 (after training set statistics): " << OUTPUT_INV_TRANSFORM(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;

	//test loading data
	Net::load_data<decltype(path)>();

	Net::discriminate();
	std::cout << "Net value with input of 1 (after load): " << OUTPUT_INV_TRANSFORM(Net::template get_layer<Net::last_layer_index>::feature_maps[0].at(0, 0)) << std::endl;

	std::cout << "\n\nPress any key to exit" << std::endl;
	_getche();
	return 0;
}