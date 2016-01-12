#include "neuralnet.h"

std::string inbetween(std::string &input, const std::string &start, const std::string &end)
{
	int i = input.find(start) + start.size();
	return input.substr(i, input.substr(i, input.size()).find(end));
}

std::vector<std::string> split(std::string &input, const std::string &delim)
{
	std::vector<std::string> result;
	int last = input.find(delim);
	std::string item = input.substr(0, last);
	if (item == "")
		return result;
	result.push_back(item);

	while (last != std::string::npos && last < input.size())
	{
		try
		{
			item = input.substr(last + delim.size(), input.substr(last + delim.size(), input.size()).find(delim));
		}
		catch (int e)
		{
			item = input.substr(last + delim.size(), input.size());
		}
		if (item != "")
			result.push_back(item);
		last += delim.size() + item.size();
	}
	return result;
}

NeuralNet::~NeuralNet()
{
	for (int i = 0; i < layers.size(); ++i)
		delete layers[i];
	for (int i = 0; i < input.size(); ++i)
		delete input[i];
	for (int i = 0; i < labels.size(); ++i)
		delete labels[i];
	for (int i = 0; i < weight_gradient.size(); ++i)
		for (int j = 0; j < weight_gradient[i].size(); ++j)
			delete weight_gradient[i][j];
	for (int i = 0; i < biases_gradient.size(); ++i)
		for (int j = 0; j < biases_gradient[i].size(); ++j)
			delete biases_gradient[i][j];
}

void NeuralNet::setup_gradient()
{
    //conditions
    if (optimization_method == CNN_OPT_ADAM)
    {
        use_momentum = false;
        use_hessian = false;
    }
    
    if (optimization_method == CNN_OPT_ADAGRAD)
    {
        use_momentum = false;
        use_hessian = false;
    }
    
	//labels and input
	input = std::vector<IMatrix<float>*>(layers[0]->feature_maps.size());
	for (int f = 0; f < input.size(); ++f)
		input[f] = layers[0]->feature_maps[f]->clone();
	labels = std::vector<IMatrix<float>*>(layers[layers.size() - 1]->feature_maps.size());
	for (int f = 0; f < labels.size(); ++f)
		labels[f] = layers[layers.size() - 1]->feature_maps[f]->clone();

	//gradients setup for weights and biases
	weight_gradient = std::vector<std::vector<IMatrix<float>*>>(layers.size());
	biases_gradient = std::vector<std::vector<IMatrix<float>*>>(layers.size());

	weight_momentum = std::vector<std::vector<IMatrix<float>*>>(layers.size());
	biases_momentum = std::vector<std::vector<IMatrix<float>*>>(layers.size());

	for (int l = 0; l < layers.size(); ++l)
	{
		weight_gradient[l] = std::vector<IMatrix<float>*>(layers[l]->recognition_weights.size());
		for (int d = 0; d < layers[l]->recognition_weights.size(); ++d)
			weight_gradient[l][d] = layers[l]->recognition_weights[d]->clone();

		if (layers[l]->use_biases)
		{
			biases_gradient[l] = std::vector<IMatrix<float>*>(layers[l]->biases.size());
			for (int f_0 = 0; f_0 < layers[l]->biases.size(); ++f_0)
				biases_gradient[l][f_0] = layers[l]->biases[f_0]->clone();
		}

		if (use_momentum || optimization_method == CNN_OPT_ADAM)
		{
			weight_momentum[l] = std::vector<IMatrix<float>*>(layers[l]->recognition_weights.size());
			for (int d = 0; d < layers[l]->recognition_weights.size(); ++d)
				weight_momentum[l][d] = layers[l]->recognition_weights[d]->clone();

			if (layers[l]->use_biases)
			{
				biases_momentum[l] = std::vector<IMatrix<float>*>(layers[l]->biases.size());
				for (int f_0 = 0; f_0 < layers[l]->biases.size(); ++f_0)
					biases_momentum[l][f_0] = layers[l]->biases[f_0]->clone();
			}
		}
	}

	//reset gradients
	for (int l = 0; l < weight_gradient.size(); ++l)
		for (int d = 0; d < weight_gradient[l].size(); ++d)
			for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					weight_gradient[l][d]->at(i, j) = 0;
	for (int l = 0; l < biases_gradient.size(); ++l)
		for (int f_0 = 0; f_0 < biases_gradient[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < biases_gradient[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < biases_gradient[l][f_0]->cols(); ++j_0)
					biases_gradient[l][f_0]->at(i_0, j_0) = 0;
	for (int l = 0; l < weight_momentum.size(); ++l)
		for (int d = 0; d < weight_momentum[l].size(); ++d)
			for (int i = 0; i < weight_momentum[l][d]->rows(); ++i)
				for (int j = 0; j < weight_momentum[l][d]->cols(); ++j)
					weight_momentum[l][d]->at(i, j) = 0;
	for (int l = 0; l < biases_momentum.size(); ++l)
		for (int f_0 = 0; f_0 < biases_momentum[l].size(); ++f_0)
			for (int i_0 = 0; i_0 < biases_momentum[l][f_0]->rows(); ++i_0)
				for (int j_0 = 0; j_0 < biases_momentum[l][f_0]->cols(); ++j_0)
					biases_momentum[l][f_0]->at(i_0, j_0) = 0;
}

void NeuralNet::save_data(std::string path)
{
	std::ofstream file(path);
	for (int l = 0; l < layers.size(); ++l)
	{
		//begin recognition_weights values
		for (int f = 0; f < layers[l]->recognition_weights.size(); ++f)
			for (int i = 0; i < layers[l]->recognition_weights[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->recognition_weights[f]->cols(); ++j)
					file << std::to_string(layers[l]->recognition_weights[f]->at(i, j)) << ',';//recognition_weights values
		//end recognition_weights values
	}

	for (int l = 0; l < layers.size(); ++l)
	{
		//begin generative_weights values
		for (int f = 0; f < layers[l]->generative_weights.size(); ++f)
			for (int i = 0; i < layers[l]->generative_weights[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->generative_weights[f]->cols(); ++j)
					file << std::to_string(layers[l]->generative_weights[f]->at(i, j)) << ',';//generative_weights values
		//end generative_weights values
	}

	for (int l = 0; l < layers.size(); ++l)
	{
		//begin biases values
		for (int f = 0; f < layers[l]->biases.size(); ++f)
			for (int i = 0; i < layers[l]->biases[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->biases[f]->cols(); ++j)
					file << std::to_string(layers[l]->biases[f]->at(i, j)) << ',';//bias values
		//end biases values
	}
	file.flush();
}

//todo: don't employ goto
void NeuralNet::load_data(std::string path)
{
	std::ifstream file(path);
	std::string data;
	int l = 0;
	int c = 0;
	int i = 0;
	int j = 0;
	bool greater = false;
	bool biases = false;
	while (std::getline(file, data, ','))
	{
		float value = std::stof(data);
		std::vector<IMatrix<float>*>* ref;
	conditions:
		if (!greater)
			ref = &layers[l]->recognition_weights;
		else if (!biases)
			ref = &layers[l]->generative_weights;
		else
			ref = &layers[l]->biases;

		if (c >= ref->size())
			goto features;

		if (j >= ref->at(c)->cols())
		{
			j = 0;
			++i;
		}

		if (i >= ref->at(c)->rows())
		{
			i = 0;
			j = 0;
			++c;
		}

		if (c >= ref->size())
		{
	features:
			i = 0; 
			j = 0;
			c = 0;
			++l;

			if (l >= layers.size())
			{
				i = 0;
				j = 0;
				c = 0;
				l = 0;
				if (!greater)
					greater = true;
				else if (!biases)
					biases = true;
				else
					return;
			}

			goto conditions;
		}
		ref->at(c)->at(i, j) = value;
		++j;
	}
}

void NeuralNet::set_input(const std::vector<IMatrix<float>*> &in)
{
	for (int f = 0; f < input.size(); ++f)
	{
		for (int i = 0; i < input[f]->rows(); ++i)
		{
			for (int j = 0; j < input[f]->cols(); ++j)
			{
				input[f]->at(i, j) = in[f]->at(i, j);
				layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);
			}
		}
	}
}

void NeuralNet::set_labels(const std::vector<IMatrix<float>*> &batch_labels)
{
	for (int f = 0; f < labels.size(); ++f)
		for (int i = 0; i < labels[f]->rows(); ++i)
			for (int j = 0; j < labels[f]->cols(); ++j)
				labels[f]->at(i, j) = batch_labels[f]->at(i, j);
}

void NeuralNet::discriminate()
{
	//reset
	for (int l = 1; l < layers.size(); ++l)
		for (int f = 0; f < layers[l]->feature_maps.size() && layers[l]->type != CNN_LAYER_INPUT; ++f)
			for (int i = 0; i < layers[l]->feature_maps[f]->rows(); ++i)
				for (int j = 0; j < layers[l]->feature_maps[f]->cols(); ++j)
					layers[l]->feature_maps[f]->at(i, j) = 0;
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		if (use_dropout && i != 0 && layers[i]->type != CNN_LAYER_SOFTMAX)
			dropout(layers[i]);
		layers[i]->feed_forwards(layers[i + 1]->feature_maps);
	}
}

std::vector<IMatrix<float>*> NeuralNet::generate()
{
	//clear output
	for (int f = 0; f < input.size(); ++f)
		for (int i = 0; i < input[f]->rows(); ++i)
			for (int j = 0; j < input[f]->cols(); ++j)
				layers[0]->feature_maps[f]->at(i, j) = 0;
	for (int l = layers.size() - 2; l >= 0; --l)
		layers[l]->feed_backwards(layers[l + 1]->feature_maps, true);
	return layers[0]->feature_maps;
}

void NeuralNet::pretrain(int iterations)
{
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i]->feed_forwards(layers[i + 1]->feature_maps);

			if (layers[i]->type == CNN_LAYER_CONVOLUTION || layers[i]->type == CNN_LAYER_PERCEPTRONFULLCONNECTIVITY)
				layers[i]->wake_sleep(learning_rate, use_dropout);
		}
	}
}

float NeuralNet::train(int iterations, float mse = 0.0f)
{
	float error;
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();
		error = global_error();

		//values of the network when fed forward
		if (error > mse / 10)
		{
			std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
			for (int l = 0; l < layers.size(); ++l)
			{
				temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
				for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
					temp[l][f] = layers[l]->feature_maps[f]->clone();
			}

			//get error signals for output and returns any layers to be skipped
			int off = error_signals();

			//backprop for each layer
			if (use_batch_learning || optimization_method == CNN_OPT_ADAM || optimization_method == CNN_OPT_ADAGRAD)
			    for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weight_gradient[l], 
						biases_gradient[l], std::vector<IMatrix<float>*>(), std::vector<IMatrix<float>*>(),
						use_hessian, minimum_divisor, false, 0.0f);
			else if (use_momentum)
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, weight_gradient[l],
					biases_gradient[l], weight_momentum[l], biases_momentum[l],
					use_hessian, minimum_divisor, true, momentum_term);
			else
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps, layers[l]->recognition_weights,
						layers[l]->biases, std::vector<IMatrix<float>*>(), std::vector<IMatrix<float>*>(),
						use_hessian, minimum_divisor, false, 0.0f);


			for (int i = 0; i < temp.size(); ++i)
				for (int j = 0; j < temp[i].size(); ++j)
					delete temp[i][j];
		}
	}
	return error;
}

float NeuralNet::train(int iterations, std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases, float mse = 0.0)
{
	float error;
	for (int e = 0; e < iterations; ++e)
	{
		//reset input
		for (int f = 0; f < input.size(); ++f)
			for (int i = 0; i < input[f]->rows(); ++i)
				for (int j = 0; j < input[f]->cols(); ++j)
					layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

		discriminate();
		error = global_error();

		if (error > mse / 10)
		{
			//values of the network when fed forward
			std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
			for (int l = 0; l < layers.size(); ++l)
			{
				temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
				for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
					temp[l][f] = layers[l]->feature_maps[f]->clone();
			}

			//get error signals for output and returns any layers to be skipped
			int off = error_signals();

			//backprop for each layer
			if (use_batch_learning || optimization_method == CNN_OPT_ADAM || optimization_method == CNN_OPT_ADAGRAD)
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps,
						weights[l], biases[l], std::vector<IMatrix<float>*>(), std::vector<IMatrix<float>*>(),
						use_hessian, minimum_divisor, false, 0.0f);
			else
				for (int l = layers.size() - 2 - off; l > 0; --l)
					layers[l]->back_prop(temp[l + 1], layers[l + 1]->feature_maps,
						weights[l], biases[l], weight_momentum[l], biases_momentum[l],
						use_hessian, minimum_divisor, use_momentum, momentum_term);

			for (int i = 0; i < temp.size(); ++i)
				for (int j = 0; j < temp[i].size(); ++j)
					delete temp[i][j];
		}
	}
	return error;
}

void NeuralNet::calculate_hessian(bool use_first_deriv, float gamma)
{
	for (int f = 0; f < input.size(); ++f)
		for (int i = 0; i < input[f]->rows(); ++i)
			for (int j = 0; j < input[f]->cols(); ++j)
				layers[0]->feature_maps[f]->at(i, j) = input[f]->at(i, j);

	discriminate();

	//values of the network when fed forward
	std::vector<std::vector<IMatrix<float>*>> temp(layers.size());
	for (int l = 0; l < layers.size(); ++l)
	{
		temp[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
		for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
			temp[l][f] = layers[l]->feature_maps[f]->clone();
	}

	//first derivatives
	std::vector<std::vector<IMatrix<float>*>> deriv_first(layers.size());
	for (int l = 0; l < layers.size() && use_first_deriv; ++l)
	{
		deriv_first[l] = std::vector<IMatrix<float>*>(layers[l]->feature_maps.size());
		for (int f = 0; f < layers[l]->feature_maps.size(); ++f)
		{
			deriv_first[l][f] = layers[l]->feature_maps[f]->clone();
			for (int i = 0; i < deriv_first[l][f]->rows(); ++i)
				for (int j = 0; j < deriv_first[l][f]->cols(); ++j)
					deriv_first[l][f]->at(i, j) = 0;
		}
	}

	//get error signals for output for first derivatives
	if (use_first_deriv)
	{
		if (cost_function == CNN_COST_QUADRATIC)
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						deriv_first[deriv_first.size() - 1][k]->at(i, j) = layers[layers.size() - 1]->feature_maps[k]->at(i, j) - labels[k]->at(i, j);
		if (cost_function == CNN_COST_CROSSENTROPY)
		{
			for (int k = 0; k < labels.size(); ++k)
			{
				for (int i = 0; i < labels[k]->rows(); ++i)
				{
					for (int j = 0; j < labels[k]->cols(); ++j)
					{
						float t = labels[k]->at(i, j);
						float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

						deriv_first[layers.size() - 1][k]->at(i, j) = (x - t) / (x * (1 - x));
					}
				}
			}
		}
		if (cost_function == CNN_COST_LOGLIKELIHOOD)
		{
			if (layers[layers.size() - 2]->type == CNN_LAYER_SOFTMAX)
				for (int k = 0; k < labels.size(); ++k)
					for (int i = 0; i < labels[k]->rows(); ++i)
						for (int j = 0; j < labels[k]->cols(); ++j)
							deriv_first[layers.size() - 2][k]->at(i, j) = layers[layers.size() - 2]->feature_maps[k]->at(i, j) - labels[k]->at(i, j);
			else
				for (int k = 0; k < labels.size(); ++k)
					for (int i = 0; i < labels[k]->rows(); ++i)
						for (int j = 0; j < labels[k]->cols(); ++j)
							deriv_first[layers.size() - 1][k]->at(i, j) = (labels[k]->at(i, j) > 0) ? -1 / layers[layers.size() - 1]->feature_maps[k]->at(i, j) : 0;
		}
	}

	//get error signals for output for second derivatives and returns any layers to be skipped
	int off = hessian_error_signals();

	//backprop for each layer
	for (int l = layers.size() - 2 - off; l > 0; --l)
		layers[l]->back_prop_second(temp[l + 1], layers[l + 1]->feature_maps, deriv_first[l + 1], deriv_first[l], use_first_deriv, gamma);

	for (int i = 0; i < temp.size(); ++i)
	{
		for (int j = 0; j < temp[i].size(); ++j)
		{
			delete temp[i][j];
			if (use_first_deriv)
				delete deriv_first[i][j];
		}
	}
}

void NeuralNet::add_layer(ILayer* layer)
{
	layers.push_back(layer);
}

void NeuralNet::apply_gradient()
{
	if (use_momentum)
	{
		//update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
		{
			for (int d = 0; d < weight_gradient[l].size(); ++d)
			{
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					{
						layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * weight_gradient[l][d]->at(i, j) 
																	+ momentum_term * weight_momentum[l][d]->at(i, j);
						weight_momentum[l][d]->at(i, j) = momentum_term * weight_momentum[l][d]->at(i, j) + weight_gradient[l][d]->at(i, j);
    					weight_gradient[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases_gradient.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases_gradient[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases_gradient[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases_gradient[l][f_0]->cols(); ++j_0)
					{
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * biases_gradient[l][f_0]->at(i_0, j_0)
																+ momentum_term * biases_momentum[l][f_0]->at(i_0, j_0);
						biases_momentum[l][f_0]->at(i_0, j_0) = momentum_term * biases_momentum[l][f_0]->at(i_0, j_0) + biases_gradient[l][f_0]->at(i_0, j_0);
    					biases_gradient[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}

    else if (optimization_method == CNN_OPT_ADAM)
    {
        t++;
        //update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
		{
			for (int d = 0; d < weight_gradient[l].size(); ++d)
			{
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					{
					    float g = weight_gradient[l][d]->at(i, j);
					    weight_momentum[l][d]->at(i, j) = beta1 * weight_momentum[l][d]->at(i, j) + (1 - beta1) * g;
					    layers[l]->hessian_weights[d]->at(i, j) = beta2 * layers[l]->hessian_weights[d]->at(i, j) + (1 - beta2) * g * g;
					    layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t)) * weight_momentum[l][d]->at(i, j) / (sqrt(layers[l]->hessian_weights[d]->at(i, j)) + epsilon); 
						weight_gradient[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases_gradient.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases_gradient[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases_gradient[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases_gradient[l][f_0]->cols(); ++j_0)
					{
						float g = biases_gradient[l][f_0]->at(i_0, j_0);
					    biases_momentum[l][f_0]->at(i_0, j_0) = beta1 * biases_momentum[l][f_0]->at(i_0, j_0) + (1 - beta1) * g;
					    layers[l]->hessian_biases[f_0]->at(i_0, j_0) = beta2 * layers[l]->hessian_biases[f_0]->at(i_0, j_0) + (1 - beta2) * g * g;
					    layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t)) * biases_momentum[l][f_0]->at(i_0, j_0) / (sqrt(layers[l]->hessian_biases[f_0]->at(i_0, j_0)) + epsilon); 
						biases_gradient[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}
	
	else if (optimization_method == CNN_OPT_ADAGRAD)
	{
	    //update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
		{
			for (int d = 0; d < weight_gradient[l].size(); ++d)
			{
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					{
					    float g = weight_gradient[l][d]->at(i, j);
						layers[l]->recognition_weights[d]->at(i, j) += learning_rate / sqrt(layers[l]->hessian_weights[d]->at(i, j)) * g;
						layers[l]->hessian_weights[d]->at(i, j) += g * g;
						weight_gradient[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases_gradient.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases_gradient[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases_gradient[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases_gradient[l][f_0]->cols(); ++j_0)
					{
					    float g = biases_gradient[l][f_0]->at(i_0, j_0);
						layers[l]->biases[f_0]->at(i_0, j_0) += learning_rate / sqrt(layers[l]->hessian_biases[f_0]->at(i_0, j_0)) * g;
						layers[l]->hessian_biases[f_0]->at(i_0, j_0) += g * g;
						biases_gradient[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}

	else
	{
		//update weights
		for (int l = 0; l < weight_gradient.size(); ++l)
		{
			for (int d = 0; d < weight_gradient[l].size(); ++d)
			{
				for (int i = 0; i < weight_gradient[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weight_gradient[l][d]->cols(); ++j)
					{
						layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * weight_gradient[l][d]->at(i, j);
						weight_gradient[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases_gradient.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases_gradient[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases_gradient[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases_gradient[l][f_0]->cols(); ++j_0)
					{
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * biases_gradient[l][f_0]->at(i_0, j_0);
						biases_gradient[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}
}

void NeuralNet::apply_gradient(std::vector<std::vector<IMatrix<float>*>> weights, std::vector<std::vector<IMatrix<float>*>> biases)
{
	if (use_momentum)
	{
		//update weights
		for (int l = 0; l < weights.size(); ++l)
		{
			for (int d = 0; d < weights[l].size(); ++d)
			{
				for (int i = 0; i < weights[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weights[l][d]->cols(); ++j)
					{
						layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * weights[l][d]->at(i, j) + momentum_term * weight_momentum[l][d]->at(i, j);
						weight_momentum[l][d]->at(i, j) = momentum_term * weight_momentum[l][d]->at(i, j) + weights[l][d]->at(i, j);
    					weights[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
					{
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * biases[l][f_0]->at(i_0, j_0) + momentum_term * biases_momentum[l][f_0]->at(i_0, j_0);
						biases_momentum[l][f_0]->at(i_0, j_0) = momentum_term * biases_momentum[l][f_0]->at(i_0, j_0) + biases[l][f_0]->at(i_0, j_0);
    					biases[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}

    else if (optimization_method == CNN_OPT_ADAM)
    {
        t++;
        //update weights
		for (int l = 0; l < weights.size(); ++l)
		{
			for (int d = 0; d < weights[l].size(); ++d)
			{
				for (int i = 0; i < weights[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weights[l][d]->cols(); ++j)
					{
					    float g = weights[l][d]->at(i, j);
					    weight_momentum[l][d]->at(i, j) = beta1 * weight_momentum[l][d]->at(i, j) + (1 - beta1) * g;
					    layers[l]->hessian_weights[d]->at(i, j) = beta2 * layers[l]->hessian_weights[d]->at(i, j) + (1 - beta2) * g * g;
					    layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t)) * weight_momentum[l][d]->at(i, j) / (sqrt(layers[l]->hessian_weights[d]->at(i, j)) + epsilon); 
						weights[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
					{
						float g = biases[l][f_0]->at(i_0, j_0);
					    biases_momentum[l][f_0]->at(i_0, j_0) = beta1 * biases_momentum[l][f_0]->at(i_0, j_0) + (1 - beta1) * g;
					    layers[l]->hessian_biases[f_0]->at(i_0, j_0) = beta2 * layers[l]->hessian_biases[f_0]->at(i_0, j_0) + (1 - beta2) * g * g;
					    layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t)) * biases_momentum[l][f_0]->at(i_0, j_0) / (sqrt(layers[l]->hessian_biases[f_0]->at(i_0, j_0)) + epsilon); 
						biases[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}

    else if (optimization_method == CNN_OPT_ADAGRAD)
	{
	    //update weights
		for (int l = 0; l < weights.size(); ++l)
		{
			for (int d = 0; d < weights[l].size(); ++d)
			{
				for (int i = 0; i < weights[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weights[l][d]->cols(); ++j)
					{
					    float g = weights[l][d]->at(i, j);
						layers[l]->recognition_weights[d]->at(i, j) += learning_rate / sqrt(layers[l]->hessian_weights[d]->at(i, j)) * g;
						layers[l]->hessian_weights[d]->at(i, j) += g * g;
						weights[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
					{
					    float g = biases[l][f_0]->at(i_0, j_0);
						layers[l]->biases[f_0]->at(i_0, j_0) += learning_rate / sqrt(layers[l]->hessian_biases[f_0]->at(i_0, j_0)) * g;
						layers[l]->hessian_biases[f_0]->at(i_0, j_0) += g * g;
						biases[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}

	else
	{
		//update weights
		for (int l = 0; l < weights.size(); ++l)
		{
			for (int d = 0; d < weights[l].size(); ++d)
			{
				for (int i = 0; i < weights[l][d]->rows(); ++i)
				{
					for (int j = 0; j < weights[l][d]->cols(); ++j)
					{
						layers[l]->recognition_weights[d]->at(i, j) += -learning_rate * weights[l][d]->at(i, j);
						weights[l][d]->at(i, j) = 0;
					}
				}
			}
		}

		//update biases
		for (int l = 0; l < biases.size(); ++l)
		{
			for (int f_0 = 0; f_0 < biases[l].size(); ++f_0)
			{
				for (int i_0 = 0; i_0 < biases[l][f_0]->rows(); ++i_0)
				{
					for (int j_0 = 0; j_0 < biases[l][f_0]->cols(); ++j_0)
					{
						layers[l]->biases[f_0]->at(i_0, j_0) += -learning_rate * biases[l][f_0]->at(i_0, j_0);
						biases[l][f_0]->at(i_0, j_0) = 0;
					}
				}
			}
		}
	}
}

float NeuralNet::global_error()
{
	if (cost_function == CNN_COST_QUADRATIC)
	{
		float sum = 0.0f;
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					sum += pow(labels[k]->at(i, j) - layers[layers.size() - 1]->feature_maps[k]->at(i, j), 2);
		return sum / 2;
	}

	else if (cost_function == CNN_COST_CROSSENTROPY)
	{
		float sum = 0.0f;
		for (int k = 0; k < labels.size(); ++k)
		{
			for (int i = 0; i < labels[k]->rows(); ++i)
			{
				for (int j = 0; j < labels[k]->cols(); ++j)
				{
					float t = labels[k]->at(i, j);
					float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

					sum += t * log(x) + (1 - t) * log(1 - x);
				}
			}
		}
		return sum;
	}

	else if (cost_function == CNN_COST_LOGLIKELIHOOD)
	{
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					if (labels[k]->at(i, j) > 0)
						return -log(layers[layers.size() - 1]->feature_maps[k]->at(i, j));
	}
}

//todo: check
int NeuralNet::error_signals()
{
	if (cost_function == CNN_COST_QUADRATIC)
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					layers[layers.size() - 1]->feature_maps[k]->at(i, j) -= labels[k]->at(i, j);
	if (cost_function == CNN_COST_CROSSENTROPY)
	{
		for (int k = 0; k < labels.size(); ++k)
		{
			for (int i = 0; i < labels[k]->rows(); ++i)
			{
				for (int j = 0; j < labels[k]->cols(); ++j)
				{
					float t = labels[k]->at(i, j);
					float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

					layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (x - t) / (x * (1 - x));
				}
			}
		}
	}
	if (cost_function == CNN_COST_LOGLIKELIHOOD)
	{
		if (layers[layers.size() - 2]->type == CNN_LAYER_SOFTMAX)
		{
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 2]->feature_maps[k]->at(i, j) -= labels[k]->at(i, j);
			return 1;
		}
		else
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (labels[k]->at(i, j) > 0) ? -1 / layers[layers.size() - 1]->feature_maps[k]->at(i, j) : 0;
	}
	if (cost_function == CNN_COST_TARGETS)
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					layers[layers.size() - 1]->feature_maps[k]->at(i, j) = labels[k]->at(i, j);
	return 0;
}

//todo: check
int NeuralNet::hessian_error_signals()
{
	if (cost_function == CNN_COST_QUADRATIC)
		for (int k = 0; k < labels.size(); ++k)
			for (int i = 0; i < labels[k]->rows(); ++i)
				for (int j = 0; j < labels[k]->cols(); ++j)
					layers[layers.size() - 1]->feature_maps[k]->at(i, j) = 1;
	if (cost_function == CNN_COST_CROSSENTROPY)
	{
		for (int k = 0; k < labels.size(); ++k)
		{
			for (int i = 0; i < labels[k]->rows(); ++i)
			{
				for (int j = 0; j < labels[k]->cols(); ++j)
				{
					float t = labels[k]->at(i, j);
					float x = layers[layers.size() - 1]->feature_maps[k]->at(i, j);

					layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (1 - (x - t) * x) / (x * (1 - x));
				}
			}
		}
	}
	if (cost_function == CNN_COST_LOGLIKELIHOOD)
	{
		if (layers[layers.size() - 2]->type == CNN_LAYER_SOFTMAX)
		{
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 2]->feature_maps[k]->at(i, j) = 1;
			return 1;
		}
		else
			for (int k = 0; k < labels.size(); ++k)
				for (int i = 0; i < labels[k]->rows(); ++i)
					for (int j = 0; j < labels[k]->cols(); ++j)
						layers[layers.size() - 1]->feature_maps[k]->at(i, j) = (labels[k]->at(i, j) > 0) ? 1 / layers[layers.size() - 1]->feature_maps[k]->at(i, j) * layers[layers.size() - 1]->feature_maps[k]->at(i, j) : 0;
	}
	return 0;
}

Matrix2D<int, 4, 1>* NeuralNet::coords(int &l, int &k, int &i, int &j)
{
	Matrix2D<int, 4, 1>* out = new Matrix2D<int, 4, 1>();
	out->at(0, 0) = l;
	out->at(1, 0) = k;
	out->at(2, 0) = i;
	out->at(3, 0) = j;
	return out;
}

void NeuralNet::dropout(ILayer* &layer)
{
	for (int f = 0; f < layer->feature_maps.size(); ++f)
		for (int i = 0; i < layer->feature_maps[f]->rows(); ++i)
			for (int j = 0; j < layer->feature_maps[f]->cols(); ++j)
				if ((1.0f * rand()) / RAND_MAX >= .5f)
					layer->feature_maps[f]->at(i, j) = 0;
}