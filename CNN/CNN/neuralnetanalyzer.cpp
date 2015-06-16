#include "neuralnetanalyzer.h"

int NeuralNetAnalyzer::sample_size = 0;
std::vector<float> NeuralNetAnalyzer::sample = std::vector<float>();

std::vector<std::vector<IMatrix<float>*>> NeuralNetAnalyzer::approximate_weight_gradient(NeuralNet &net)
{
	//setup output
	std::vector<std::vector<IMatrix<float>*>> output(net.layers.size());
	for (int i = 0; i < output.size(); ++i)
	{
		output[i] = std::vector<IMatrix<float>*>(net.layers[i]->recognition_data.size());
		for (int j = 0; j < output[i].size(); ++j)
			output[i][j] = net.layers[i]->recognition_data[j]->clone();
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one weight at a time, this requires the network to be ran for every single weight
	for (int l = 0; l < net.layers.size(); ++l)
	{
		for (int d = 0; d < net.layers[l]->recognition_data.size(); ++d)
		{
			for (int i = 0; i < net.layers[l]->recognition_data[d]->rows(); ++i)
			{
				for (int j = 0; j < net.layers[l]->recognition_data[d]->cols(); ++j)
				{
					//adjust current weight
					net.layers[l]->recognition_data[d]->at(i, j) += .001f;

					//evaluate network with adjusted weight and approximate derivative
					net.discriminate();
					float adjusted_error = net.global_error();
					output[l][d]->at(i, j) = (adjusted_error - original_error) / .001f;

					//reset weight
					net.layers[l]->recognition_data[d]->at(i, j) -= .001f;
				}
			}
		}
	}

	return output;
}

std::vector<std::vector<IMatrix<float>*>> NeuralNetAnalyzer::approximate_bias_gradient(NeuralNet &net)
{
	//setup output
	std::vector<std::vector<IMatrix<float>*>> output(net.layers.size());
	for (int i = 0; i < output.size(); ++i)
	{
		if (net.layers[i]->use_biases)
		{
			output[i] = std::vector<IMatrix<float>*>(net.layers[i]->biases.size());
			for (int f = 0; f < output[i].size(); ++f)
				output[i][f] = net.layers[i]->biases[f]->clone();
		}
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one bias at a time, this requires the network to be ran for every single bias
	for (int l = 0; l < net.layers.size(); ++l)
	{
		if (net.layers[l]->use_biases)
		{
			for (int f = 0; f < net.layers[l]->biases.size(); ++f)
			{
				for (int i = 0; i < net.layers[l]->biases[f]->rows(); ++i)
				{
					for (int j = 0; j < net.layers[l]->biases[f]->cols(); ++j)
					{
						//adjust current weight
						net.layers[l]->biases[f]->at(i, j) += .001f;

						//evaluate network with adjusted weight and approximate derivative
						net.discriminate();
						float adjusted_error = net.global_error();
						output[l][f]->at(i, j) = (adjusted_error - original_error) / .001f;

						//reset weight
						net.layers[l]->biases[f]->at(i, j) -= .001f;
					}
				}
			}
		}
	}

	return output;
}

void NeuralNetAnalyzer::add_point(float value)
{
	if (sample.size() == sample_size)
		sample.erase(sample.begin());
	sample.push_back(value);
}

float NeuralNetAnalyzer::mean_squared_error()
{
	float sum = 0.0f;
	for (int i = 0; i < sample.size(); ++i)
		sum += sample[i];
	return sum / sample.size();
}