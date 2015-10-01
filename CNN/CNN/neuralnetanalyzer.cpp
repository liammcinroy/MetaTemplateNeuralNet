#include "neuralnetanalyzer.h"

int NeuralNetAnalyzer::sample_size = 0;
std::vector<float> NeuralNetAnalyzer::sample = std::vector<float>();
std::vector<float> NeuralNetAnalyzer::mses = std::vector<float>();


std::vector<std::vector<IMatrix<float>*>> NeuralNetAnalyzer::approximate_weight_gradient(NeuralNet &net)
{
	//setup output
	std::vector<std::vector<IMatrix<float>*>> output(net.layers.size());
	for (int i = 0; i < output.size(); ++i)
	{
		output[i] = std::vector<IMatrix<float>*>(net.layers[i]->recognition_weights.size());
		for (int j = 0; j < output[i].size(); ++j)
			output[i][j] = net.layers[i]->recognition_weights[j]->clone();
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one weight at a time, this requires the network to be ran for every single weight
	for (int l = 0; l < net.layers.size(); ++l)
	{
		for (int d = 0; d < net.layers[l]->recognition_weights.size(); ++d)
		{
			for (int i = 0; i < net.layers[l]->recognition_weights[d]->rows(); ++i)
			{
				for (int j = 0; j < net.layers[l]->recognition_weights[d]->cols(); ++j)
				{
					//adjust current weight
					net.layers[l]->recognition_weights[d]->at(i, j) += .001f;

					//evaluate network with adjusted weight and approximate derivative
					net.discriminate();
					float adjusted_error = net.global_error();
					output[l][d]->at(i, j) = -net.learning_rate * (adjusted_error - original_error) / .001f;

					//reset weight
					net.layers[l]->recognition_weights[d]->at(i, j) -= .001f;
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
		output[i] = std::vector<IMatrix<float>*>(net.layers[i]->biases.size());
		for (int f = 0; f < output[i].size(); ++f)
			output[i][f] = net.layers[i]->biases[f]->clone();
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one bias at a time, this requires the network to be ran for every single bias
	for (int l = 0; l < net.layers.size(); ++l)
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
					float diff = -net.learning_rate * (adjusted_error - original_error) / .001f;
					output[l][f]->at(i, j) = diff;

					//reset weight
					net.layers[l]->biases[f]->at(i, j) -= .001f;
				}
			}
		}
	}

	return output;
}

std::vector<std::vector<IMatrix<float>*>> NeuralNetAnalyzer::approximate_weight_hessian(NeuralNet &net)
{
	//setup output
	std::vector<std::vector<IMatrix<float>*>> output(net.layers.size());
	for (int i = 0; i < output.size(); ++i)
	{
		output[i] = std::vector<IMatrix<float>*>(net.layers[i]->recognition_weights.size());
		for (int j = 0; j < output[i].size(); ++j)
			output[i][j] = net.layers[i]->recognition_weights[j]->clone();
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one weight at a time, this requires the network to be ran for every single weight
	for (int l = 0; l < net.layers.size(); ++l)
	{
		for (int d = 0; d < net.layers[l]->recognition_weights.size(); ++d)
		{
			for (int i = 0; i < net.layers[l]->recognition_weights[d]->rows(); ++i)
			{
				for (int j = 0; j < net.layers[l]->recognition_weights[d]->cols(); ++j)
				{
					//adjust current weight
					net.layers[l]->recognition_weights[d]->at(i, j) -= .001f;

					//evaluate network with adjusted weight
					net.discriminate();
					float h_minus = net.global_error();

					//adjust current weight
					net.layers[l]->recognition_weights[d]->at(i, j) += .002f;

					//evaluate network with adjusted weight
					net.discriminate();
					float h = net.global_error();

					//approximate with derivative
					output[l][d]->at(i, j) = (h - 2 * original_error + h_minus) / (.001f * .001f);

					//reset weight
					net.layers[l]->recognition_weights[d]->at(i, j) -= .001f;
				}
			}
		}
	}

	return output;
}

std::vector<std::vector<IMatrix<float>*>> NeuralNetAnalyzer::approximate_bias_hessian(NeuralNet &net)
{
	//setup output
	std::vector<std::vector<IMatrix<float>*>> output(net.layers.size());
	for (int i = 0; i < output.size(); ++i)
	{
		output[i] = std::vector<IMatrix<float>*>(net.layers[i]->biases.size());
		for (int j = 0; j < output[i].size(); ++j)
			output[i][j] = net.layers[i]->biases[j]->clone();
	}

	//find error for current network
	net.discriminate();
	float original_error = net.global_error();

	//begin evaluating derivatives of the error numerically for only one weight at a time, this requires the network to be ran for every single weight
	for (int l = 0; l < net.layers.size(); ++l)
	{
		for (int d = 0; d < net.layers[l]->biases.size(); ++d)
		{
			for (int i = 0; i < net.layers[l]->biases[d]->rows(); ++i)
			{
				for (int j = 0; j < net.layers[l]->biases[d]->cols(); ++j)
				{
					//adjust current weight
					net.layers[l]->biases[d]->at(i, j) -= .001f;

					//evaluate network with adjusted weight
					net.discriminate();
					float h_minus = net.global_error();

					//adjust current weight
					net.layers[l]->biases[d]->at(i, j) += .002f;

					//evaluate network with adjusted weight
					net.discriminate();
					float h = net.global_error();

					//approximate with derivative
					output[l][d]->at(i, j) = (h - 2 * original_error + h_minus) / (.001f * .001f);

					//reset weight
					net.layers[l]->biases[d]->at(i, j) -= .001f;
				}
			}
		}
	}

	return output;
}

std::pair<float, float> NeuralNetAnalyzer::mean_gradient_error(NeuralNet &net, std::vector<std::vector<IMatrix<float>*>> observed_weights,
	std::vector<std::vector<IMatrix<float>*>> observed_biases)
{
	std::vector<std::vector<IMatrix<float>*>> expected_weights = NeuralNetAnalyzer::approximate_weight_gradient(net);
	std::vector<std::vector<IMatrix<float>*>> expected_biases = NeuralNetAnalyzer::approximate_bias_gradient(net);

	float weight_sum = 0.0f;
	float bias_sum = 0.0f;

	int weight_n = 0;
	int bias_n = 0;

	for (int l = 0; l < expected_weights.size(); ++l)
	{
		for (int d = 0; d < expected_weights[l].size(); ++d)
		{
			for (int i = 0; i < expected_weights[l][d]->rows(); ++i)
			{
				for (int j = 0; j < expected_weights[l][d]->cols(); ++j)
				{
					weight_sum += abs(expected_weights[l][d]->at(i, j) - observed_weights[l][d]->at(i, j));
					++weight_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_biases.size(); ++l)
	{
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
		{
			for (int i_0 = 0; i_0 < expected_biases[l][f_0]->rows(); ++i_0)
			{
				for (int j_0 = 0; j_0 < expected_biases[l][f_0]->cols(); ++j_0)
				{
					bias_sum += abs(expected_biases[l][f_0]->at(i_0, j_0) - observed_biases[l][f_0]->at(i_0, j_0));
					++bias_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_weights.size(); ++l)
		for (int d = 0; d < expected_weights[l].size(); ++d)
			delete expected_weights[l][d];
	for (int l = 0; l < expected_biases.size(); ++l)
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
			delete expected_biases[l][f_0];

	return std::make_pair(weight_sum / weight_n, bias_sum / bias_n);
}

std::pair<float, float> NeuralNetAnalyzer::mean_hessian_error(NeuralNet &net)
{
	std::vector<std::vector<IMatrix<float>*>> expected_weights = NeuralNetAnalyzer::approximate_weight_hessian(net);
	std::vector<std::vector<IMatrix<float>*>> expected_biases = NeuralNetAnalyzer::approximate_bias_hessian(net);

	net.calculate_hessian(true, 1);

	float weight_sum = 0.0f;
	float bias_sum = 0.0f;

	int weight_n = 0;
	int bias_n = 0;

	for (int l = 0; l < expected_weights.size(); ++l)
	{
		for (int d = 0; d < expected_weights[l].size(); ++d)
		{
			for (int i = 0; i < expected_weights[l][d]->rows(); ++i)
			{
				for (int j = 0; j < expected_weights[l][d]->cols(); ++j)
				{
					weight_sum += abs(expected_weights[l][d]->at(i, j) - net.layers[l]->hessian_weights[d]->at(i, j));
					++weight_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_biases.size(); ++l)
	{
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
		{
			for (int i_0 = 0; i_0 < expected_biases[l][f_0]->rows(); ++i_0)
			{
				for (int j_0 = 0; j_0 < expected_biases[l][f_0]->cols(); ++j_0)
				{
					bias_sum += abs(expected_biases[l][f_0]->at(i_0, j_0) - net.layers[l]->hessian_biases[f_0]->at(i_0, j_0));
					++bias_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_weights.size(); ++l)
		for (int d = 0; d < expected_weights[l].size(); ++d)
			delete expected_weights[l][d];
	for (int l = 0; l < expected_biases.size(); ++l)
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
			delete expected_biases[l][f_0];

	return std::make_pair(weight_sum / weight_n, bias_sum / bias_n);
}

std::pair<float, float> NeuralNetAnalyzer::proportional_gradient_error(NeuralNet &net, std::vector<std::vector<IMatrix<float>*>> observed_weights,
	std::vector<std::vector<IMatrix<float>*>> observed_biases)
{
	std::vector<std::vector<IMatrix<float>*>> expected_weights = NeuralNetAnalyzer::approximate_weight_gradient(net);
	std::vector<std::vector<IMatrix<float>*>> expected_biases = NeuralNetAnalyzer::approximate_bias_gradient(net);

	float weight_sum = 0.0f;
	float bias_sum = 0.0f;

	int weight_n = 0;
	int bias_n = 0;

	for (int l = 0; l < expected_weights.size(); ++l)
	{
		for (int d = 0; d < expected_weights[l].size(); ++d)
		{
			for (int i = 0; i < expected_weights[l][d]->rows(); ++i)
			{
				for (int j = 0; j < expected_weights[l][d]->cols(); ++j)
				{
					weight_sum += abs(expected_weights[l][d]->at(i, j) - observed_weights[l][d]->at(i, j)) / observed_weights[l][d]->at(i, j);
					++weight_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_biases.size(); ++l)
	{
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
		{
			for (int i_0 = 0; i_0 < expected_biases[l][f_0]->rows(); ++i_0)
			{
				for (int j_0 = 0; j_0 < expected_biases[l][f_0]->cols(); ++j_0)
				{
					bias_sum += abs(expected_biases[l][f_0]->at(i_0, j_0) - observed_biases[l][f_0]->at(i_0, j_0)) / observed_biases[l][f_0]->at(i_0, j_0);
					++bias_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_weights.size(); ++l)
		for (int d = 0; d < expected_weights[l].size(); ++d)
			delete expected_weights[l][d];
	for (int l = 0; l < expected_biases.size(); ++l)
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
			delete expected_biases[l][f_0];

	return std::make_pair(weight_sum / weight_n, bias_sum / bias_n);
}

std::pair<float, float> NeuralNetAnalyzer::proportional_hessian_error(NeuralNet &net)
{
	std::vector<std::vector<IMatrix<float>*>> expected_weights = NeuralNetAnalyzer::approximate_weight_hessian(net);
	std::vector<std::vector<IMatrix<float>*>> expected_biases = NeuralNetAnalyzer::approximate_bias_hessian(net);

	net.calculate_hessian(true, 1);

	float weight_sum = 0.0f;
	float bias_sum = 0.0f;

	int weight_n = 0;
	int bias_n = 0;

	for (int l = 0; l < expected_weights.size(); ++l)
	{
		for (int d = 0; d < expected_weights[l].size(); ++d)
		{
			for (int i = 0; i < expected_weights[l][d]->rows(); ++i)
			{
				for (int j = 0; j < expected_weights[l][d]->cols(); ++j)
				{
					weight_sum += abs(expected_weights[l][d]->at(i, j) - net.layers[l]->hessian_weights[d]->at(i, j)) / net.layers[l]->hessian_weights[d]->at(i, j);
					++weight_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_biases.size(); ++l)
	{
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
		{
			for (int i_0 = 0; i_0 < expected_biases[l][f_0]->rows(); ++i_0)
			{
				for (int j_0 = 0; j_0 < expected_biases[l][f_0]->cols(); ++j_0)
				{
					bias_sum += abs(expected_biases[l][f_0]->at(i_0, j_0) - net.layers[l]->hessian_biases[f_0]->at(i_0, j_0)) / net.layers[l]->hessian_biases[f_0]->at(i_0, j_0);
					++bias_n;
				}
			}
		}
	}

	for (int l = 0; l < expected_weights.size(); ++l)
		for (int d = 0; d < expected_weights[l].size(); ++d)
			delete expected_weights[l][d];
	for (int l = 0; l < expected_biases.size(); ++l)
		for (int f_0 = 0; f_0 < expected_biases[l].size(); ++f_0)
			delete expected_biases[l][f_0];

	return std::make_pair(weight_sum / weight_n, bias_sum / bias_n);
}

void NeuralNetAnalyzer::save_mean_square_error(std::string path)
{
	std::ofstream file(path);
	for (int i = 0; i < mses.size(); ++i)
		file << std::to_string(mses[i]) << ',';
	file.flush();
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
	mses.push_back(sum / sample.size());
	return sum / sample.size();
}