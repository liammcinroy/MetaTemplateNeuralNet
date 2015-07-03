#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_INPUT 0
#define CNN_CONVOLUTION 1
#define CNN_PERCEPTRON_FULL_CONNECTIVITY 2
#define CNN_PERCEPTRON_LOCAL_CONNECTIVITY 3
#define CNN_MAXPOOL 4
#define CNN_SOFTMAX 5
#define CNN_OUTPUT 6

#define CNN_LINEAR 0
#define CNN_SIGMOID 1
#define CNN_BIPOLAR_SIGMOID 2
#define CNN_TANH 3
#define CNN_RELU 4

template<int rows, int cols, int kernel_rows, int kernel_cols, int stride> IMatrix<float>*
convolve(IMatrix<float>* &input, IMatrix<float>* &kernel)
{
	const int N = (kernel_rows - 1) / 2;
	const int M = (kernel_cols - 1) / 2;
	const int out_rows = (rows - kernel_rows) / stride + 1;
	const int out_cols = (cols - kernel_cols) / stride + 1;
	Matrix2D<float, out_rows, out_cols>* output = new Matrix2D<float, out_rows, out_cols>();

	//change focus of kernel
	for (int i = N; i < (rows - N); i += stride)
	{
		for (int j = M; j < (cols - M); j += stride)
		{
			//iterate over kernel
			float sum = 0;
			for (int n = N; n >= -N; --n)
				for (int m = M; m >= -M; --m)
					sum += input->at(i - n, j - m) * kernel->at(N - n, N - m);
			output->at((i - N) / stride, (j - N) / stride) = sum;
		}
	}
	return output;
}

template<int rows, int cols, int kernel_rows, int kernel_cols, int stride> void
backprop_kernel(IMatrix<float>* &input, IMatrix<float>* &output, IMatrix<float>* &kernel_gradient, float learning_rate)
{
	const int N = (kernel_rows - 1) / 2;
	const int M = (kernel_cols - 1) / 2;
	const int out_rows = (rows - kernel_rows) / stride + 1;
	const int out_cols = (cols - kernel_cols) / stride + 1;

	//change focus of kernel
	for (int i = N; i < (rows - N); i += stride)
	{
		for (int j = M; j < (cols - M); j += stride)
		{
			//iterate over kernel
			float sum = 0;
			float out = output->at((i - N) / stride, (j - N) / stride);
			for (int n = N; n >= -N; --n)
				for (int m = M; m >= -M; --m)
					kernel_gradient->at(N - n, M - m) += -learning_rate * input->at(i - n, j - m) * out;
		}
	}
}

template<int rows, int cols, int kernel_rows, int kernel_cols, int stride> IMatrix<float>*
convolve_back(IMatrix<float>* &input, IMatrix<float>* &kernel)
{
	const int N = (kernel_rows - 1) / 2;
	const int M = (kernel_cols - 1) / 2;
	Matrix2D<float, rows, cols>* output = new Matrix2D<float, rows, cols>();

	int times_across = 0;
	int times_down = 0;

	for (int i = N; i < (rows - N); i += stride)
	{
		for (int j = M; j < (cols - M); j += stride)
		{
			//find all possible ways convolved into
			float sum = 0;
			for (int n = N; n >= -N; --n)
			{
				for (int m = M; m >= -M; --m)
				{
					output->at(i - n, j - m) += kernel->at(N - n, M - m) * input->at(times_down, times_across);
				}
			}
			++times_across;
		}
		times_across = 0;
		++times_down;
	}
	return output;
}

class  ILayer
{
public:
	ILayer() = default;

	virtual ~ILayer() = default;

	virtual void feed_forwards(std::vector<IMatrix<float>*> &output) = 0;

	virtual void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights) = 0;

	virtual void wake_sleep(float &learning_rate, bool &use_dropout) = 0;

	virtual  void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate) = 0;

	virtual ILayer* clone() = 0;

	std::vector<IMatrix<float>*> feature_maps;

	std::vector<IMatrix<float>*> biases;

	std::vector<IMatrix<float>*> recognition_data;

	std::vector<IMatrix<float>*> generative_data;

	std::vector<IMatrix<float>*> second_derivatives;

	int type;

	int connections_per_neuron;

	bool use_biases = true;

	int activation = 0;

	inline float activate(float value, int activation)
	{
		if (activation == CNN_LINEAR)
			return value;
		if (activation == CNN_SIGMOID)
			return value < 5 && value > -5 ? (1 / (1 + exp(-value))) : (value >= 5 ? 1.0f : 0.0f);
		if (activation == CNN_BIPOLAR_SIGMOID)
			return value < 5 && value > -5 ? ((2 / (1 + exp(-value))) - 1) : (value >= 5 ? 1.0f : -1.0f);
		if (activation == CNN_TANH)
			return value < 5 && value > -5 ? tanh(value) : (value >= 5 ? 1.0f : -1.0f);
		if (activation == CNN_RELU)
			return value > 0 ? value : 0;
	}

	inline float activation_derivative(float value, int activation)
	{
		if (activation == CNN_LINEAR)
			return 1;
		if (activation == CNN_SIGMOID)
			return value * (1 - value);
		if (activation == CNN_BIPOLAR_SIGMOID)
			return (1 + value) * (1 - value) / 2;
		if (activation == CNN_TANH)
			return 1 - value * value;
		if (activation == CNN_RELU)
			return value > 0 ? 1 : 0;
	}
};

template<int features, int rows, int cols,
	int kernel_size, int stride, int out_features, int activation_function>
class ConvolutionLayer : public ILayer
{
public:
	ConvolutionLayer<features, rows, cols, kernel_size, stride, out_features, activation_function>()
	{
		activation = activation_function;
		type = CNN_CONVOLUTION;
		feature_maps = std::vector<IMatrix<float>*>(features);
		for (int f = 0; f < features; ++f)
			feature_maps[f] = new Matrix2D<float, rows, cols>();

		biases = std::vector<IMatrix<float>*>(out_features * features);
		recognition_data = std::vector<IMatrix<float>*>(out_features * features);
		generative_data = std::vector<IMatrix<float>*>(out_features * features);
		for (int k = 0; k < out_features * features; ++k)
		{
			biases[k] = new Matrix2D<float, 1, 1>({ 0 });
			recognition_data[k] = new Matrix2D<float, kernel_size, kernel_size>();
			generative_data[k] = new Matrix2D<float, kernel_size, kernel_size>();
			for (int i = 0; i < kernel_size; ++i)
			{
				for (int j = 0; j < kernel_size; ++j)
				{
					//purely random works best
					recognition_data[k]->at(i, j) = .05f * ((2.0f * rand()) / RAND_MAX - 1);
					generative_data[k]->at(i, j) = recognition_data[k]->at(i, j);
				}
			}
		}
	}

	~ConvolutionLayer<features, rows, cols, kernel_size, stride, out_features, activation_function>()
	{
		for (int i = 0; i < features; ++i)
			delete feature_maps[i];
		for (int i = 0; i < out_features; ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		const int out_rows = (rows - kernel_size) / stride + 1;
		const int out_cols = (cols - kernel_size) / stride + 1;

		for (int f_0 = 0; f_0 < out_features; ++f_0)
		{
			//sum the kernels
			for (int f = 0; f < features; ++f)
			{
				add<float, out_rows, out_cols>
					(output[f_0], convolve<rows, cols, kernel_size, kernel_size, stride>(feature_maps[f], recognition_data[f_0 * features + f]));
				if (use_biases)
					for (int i = 0; i < out_rows; ++i)
						for (int j = 0; j < out_cols; ++j)
							output[f_0]->at(i, j) += biases[f_0 * features + f]->at(0, 0);
			}

			if (activation != CNN_LINEAR)
			{
				for (int i = 0; i < out_rows; ++i)
					for (int j = 0; j < out_cols; ++j)
						output[f_0]->at(i, j) = activate(output[f_0]->at(i, j), activation);
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//Do the first only
		for (int f = 0; f < features; ++f)
		{
			for (int f_0 = 0; f_0 < out_features; ++f_0)
			{
				if (!use_g_weights)
					add<float, rows, cols>(feature_maps[f],
					convolve_back<rows, cols, kernel_size, kernel_size, stride>(input[f_0], recognition_data[f_0 * features + f]));
				else
					add<float, rows, cols>(feature_maps[f],
					convolve_back<rows, cols, kernel_size, kernel_size, stride>(input[f_0], generative_data[f_0 * features + f]));
			}
		}
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
		//find difference via gibbs sampling
		std::vector<IMatrix<float>*> discriminated(out_features);
		for (int i = 0; i < out_features; ++i)
			discriminated[i] = new Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>();

		std::vector<IMatrix<float>*> reconstructed(out_features);
		for (int i = 0; i < out_features; ++i)
			reconstructed[i] = new Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>();

		this->feed_forwards(discriminated);
		this->feed_backwards(discriminated, true);
		this->feed_forwards(reconstructed);

		//adjust weights
		for (int f_0 = 0; f_0 < reconstructed.size(); ++f_0)
		{
			int N = (generative_data[f_0]->rows() - 1) / 2;
			int times_down = 0;
			int times_across = 0;

			for (int i = N; i < (rows - generative_data[f_0]->rows()) + stride; i += stride)
			{
				for (int j = N; j < (cols - generative_data[f_0]->rows()) + stride; j += stride)
				{
					float delta_w = -learning_rate *
						(discriminated[f_0]->at(times_across, times_down) - reconstructed[f_0]->at(times_across, times_down));

					for (int n = N; n >= -N; --n)
					{
						for (int m = N; m >= -N; --m)
						{
							recognition_data[f_0]->at(N - n, N - m) += delta_w;
							generative_data[f_0]->at(N - n, N - m) -= delta_w;
						}
					}
					++times_across;
				}
				times_across = 0;
				++times_down;
			}
		}

		for (int i = 0; i < out_features; ++i)
		{
			delete reconstructed[i];
			delete discriminated[i];
		}
	}

	//todo: fix
	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
		std::vector<IMatrix<float>*> temp = std::vector<IMatrix<float>*>(features);
		for (int f = 0; f < features; ++f)
		{
			temp[f] = feature_maps[f]->clone();
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					feature_maps[f]->at(i, j) = 0;
		}

		const int out_rows = (rows - kernel_size) / stride + 1;
		const int out_cols = (cols - kernel_size) / stride + 1;

		//adjust gradients and update features
		for (int f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (int i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (int j_0 = 0; j_0 < out_cols; ++j_0)
				{
					deriv[f_0]->at(i_0, j_0) *= activation_derivative(data[f_0]->at(i_0, j_0), activation);
				}
			}

			for (int f = 0; f < features; ++f)
			{
				//adjust the gradient
				backprop_kernel<rows, cols, kernel_size, kernel_size, stride>(temp[f], deriv[f_0], weight_gradient[f_0 * features + f], learning_rate);

				if (use_biases)
					for (int i_0 = 0; i_0 < out_rows; ++i_0)
						for (int j_0 = 0; j_0 < out_cols; ++j_0)
							bias_gradient[f_0 * features + f]->at(0, 0) += -learning_rate * deriv[f_0]->at(i_0, j_0);
			}
		}

		//update deltas
		feed_backwards(deriv, false);

		//clean up
		for (int f = 0; f < features; ++f)
			delete temp[f];
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new ConvolutionLayer<features, rows, cols, kernel_size, stride, out_features, activation_function>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		for (int f = 0; f < biases.size(); ++f)
			for (int i = 0; i < biases[f]->rows(); ++i)
				for (int j = 0; j < biases[f]->cols(); ++j)
					copy->biases[f]->at(i, j) = this->biases[f]->at(i, j);

		for (int d = 0; d < recognition_data.size(); ++d)
			for (int i = 0; i < recognition_data[d]->rows(); ++i)
				for (int j = 0; j < recognition_data[d]->cols(); ++j)
					copy->recognition_data[d]->at(i, j) = this->recognition_data[d]->at(i, j);

		for (int d = 0; d < generative_data.size(); ++d)
			for (int i = 0; i < generative_data[d]->rows(); ++i)
				for (int j = 0; j < generative_data[d]->cols(); ++j)
					copy->generative_data[d]->at(i, j) = this->generative_data[d]->at(i, j);

		for (int d = 0; d < second_derivatives.size(); ++d)
			for (int i = 0; i < second_derivatives[d]->rows(); ++i)
				for (int j = 0; j < second_derivatives[d]->cols(); ++j)
					copy->second_derivatives[d]->at(i, j) = this->second_derivatives[d]->at(i, j);

		return copy;
	}
};

template<int features, int rows, int cols, int out_features, int out_rows, int out_cols, int activation_function>
class PerceptronFullConnectivityLayer : public ILayer
{
public:
	PerceptronFullConnectivityLayer<features, rows, cols, out_features, out_rows, out_cols, activation_function>()
	{
		activation = activation_function;
		type = CNN_PERCEPTRON_FULL_CONNECTIVITY;
		feature_maps = std::vector<IMatrix<float>*>(features);
		biases = std::vector<IMatrix<float>*>(out_features);
		recognition_data = std::vector<IMatrix<float>*>(1);
		generative_data = std::vector<IMatrix<float>*>(1);
		second_derivatives = std::vector<IMatrix<float>*>(1);

		for (int k = 0; k < features; ++k)
			feature_maps[k] = new Matrix2D<float, rows, cols>();

		for (int k = 0; k < out_features; ++k)
			biases[k] = new Matrix2D<float, out_rows, out_cols>();

		//const float s_d = 1.0f / (rows * cols * features);
		const float beta = .7 * pow(out_rows * out_cols * out_features, 1 / (rows * cols * features));
		float n = 0.0f;

		recognition_data[0] = new Matrix2D<float, out_rows * out_cols * out_features, rows * cols * features>();
		generative_data[0] = new Matrix2D<float, out_rows * out_cols * out_features, rows * cols * features>();
		second_derivatives[0] = new Matrix2D<float, out_rows * out_cols * out_features, rows * cols * features>();
		for (int i = 0; i < out_rows * out_cols * out_features; ++i)
		{
			for (int j = 0; j < rows * cols * features; ++j)
			{
				//gaussian distributed
				float x = sqrt(-2 * log(1.0f * (rand() + 1) / (RAND_MAX + 1))) * sin(2 * 3.14152f * rand() / RAND_MAX);
				recognition_data[0]->at(i, j) = x * .1f;
				generative_data[0]->at(i, j) = recognition_data[0]->at(i, j);
				second_derivatives[0]->at(i, j) = 0;
			}
		}

		float factor = beta / sqrt(n);
	}

	~PerceptronFullConnectivityLayer<features, rows, cols, out_features, out_rows, out_cols, activation_function>()
	{
		delete recognition_data[0];
		delete generative_data[0];
		delete second_derivatives[0];
		for (int i = 0; i < features; ++i)
			delete feature_maps[i];
		for (int i = 0; i < out_features; ++i)
			delete biases[i];
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		//loop through every neuron in output
		for (int f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i_0 = 0; i_0 < out_rows; ++i_0)
				{
					for (int j_0 = 0; j_0 < out_cols; ++j_0)
					{
						//loop through every neuron in input and add it to output
						float sum = 0.0f;
						for (int i = 0; i < rows; ++i)
							for (int j = 0; j < cols; ++j)
								sum += (feature_maps[f]->at(i, j) *
								recognition_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j));

						//add bias
						if (use_biases)
							output[f_0]->at(i_0, j_0) = activate(sum + biases[f_0]->at(i_0, j_0), activation);
						else
							output[f_0]->at(i_0, j_0) = activate(sum, activation);
					}
				}
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//go through every neuron in this layer
		for (int f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < rows; ++i)
				{
					for (int j = 0; j < cols; ++j)
					{
						//go through every neuron in output layer and add it to this neuron
						float sum = 0.0f;
						for (int i_0 = 0; i_0 < out_rows; ++i_0)
						{
							for (int j_0 = 0; j_0 < out_cols; ++j_0)
							{
								if (use_g_weights)
									sum += generative_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j)
									* input[f_0]->at(i_0, j_0);
								else
									sum += recognition_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j)
									* input[f_0]->at(i_0, j_0);
							}
						}
						feature_maps[f]->at(i, j) = sum;
					}
				}
			}
		}
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
		//find difference via gibbs sampling
		std::vector<IMatrix<float>*> discriminated(out_features);
		for (int i = 0; i < out_features; ++i)
			discriminated[i] = new Matrix2D<float, out_rows, out_cols>();

		std::vector<IMatrix<float>*> reconstructed(out_features);
		for (int i = 0; i < out_features; ++i)
			reconstructed[i] = new Matrix2D<float, out_rows, out_cols>();

		this->feed_forwards(discriminated);
		this->feed_backwards(discriminated, true);
		this->feed_forwards(reconstructed);

		//adjust weights
		for (int f_0 = 0; f_0 < reconstructed.size(); ++f_0)
		{
			for (int i_0 = 0; i_0 < reconstructed[f_0]->rows(); ++i_0)
			{
				for (int j_0 = 0; j_0 < reconstructed[f_0]->cols(); ++j_0)
				{
					float delta_weight = -learning_rate * (discriminated[f_0]->at(i_0, 0) - reconstructed[f_0]->at(i_0, 0));
					for (int f = 0; f < feature_maps.size(); ++f)
					{
						for (int i = 0; i < feature_maps[f]->rows(); ++i)
						{
							for (int j = 0; j < feature_maps[f]->cols(); ++j)
							{
								recognition_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += delta_weight;
								generative_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) -= delta_weight;
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < reconstructed.size(); ++i)
			delete reconstructed[i];
		for (int i = 0; i < discriminated.size(); ++i)
			delete discriminated[i];
	}

	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
		std::vector<IMatrix<float>*> temp = std::vector<IMatrix<float>*>(features);
		for (int f = 0; f < features; ++f)
		{
			temp[f] = feature_maps[f]->clone();
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					feature_maps[f]->at(i, j) = 0;
		}

		for (int f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (int i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (int j_0 = 0; j_0 < out_cols; ++j_0)
				{
					deriv[f_0]->at(i_0, j_0) *= activation_derivative(data[f_0]->at(i_0, j_0), activation);

					if (use_biases)
						bias_gradient[f_0]->at(i_0, j_0) += -learning_rate * deriv[f_0]->at(i_0, j_0);

					for (int f = 0; f < features; ++f)
					{
						for (int i = 0; i < rows; ++i)
						{
							for (int j = 0; j < cols; ++j)
							{
								feature_maps[f]->at(i, j) += 
									deriv[f_0]->at(i_0, j_0) * recognition_data[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);
								weight_gradient[0]->at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) +=
									-learning_rate * deriv[f_0]->at(i_0, j_0) * temp[f]->at(i, j);
							}
						}
					}
				}
			}
		}

		for (int f = 0; f < features; ++f)
			delete temp[f];
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new PerceptronFullConnectivityLayer<features, rows, cols, out_rows, out_cols, out_features, activation_function>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		for (int f = 0; f < biases.size(); ++f)
			for (int i = 0; i < biases[f]->rows(); ++i)
				for (int j = 0; j < biases[f]->cols(); ++j)
					copy->biases[f]->at(i, j) = this->biases[f]->at(i, j);

		for (int d = 0; d < recognition_data.size(); ++d)
			for (int i = 0; i < recognition_data[d]->rows(); ++i)
				for (int j = 0; j < recognition_data[d]->cols(); ++j)
					copy->recognition_data[d]->at(i, j) = this->recognition_data[d]->at(i, j);

		for (int d = 0; d < generative_data.size(); ++d)
			for (int i = 0; i < generative_data[d]->rows(); ++i)
				for (int j = 0; j < generative_data[d]->cols(); ++j)
					copy->generative_data[d]->at(i, j) = this->generative_data[d]->at(i, j);

		for (int d = 0; d < second_derivatives.size(); ++d)
			for (int i = 0; i < second_derivatives[d]->rows(); ++i)
				for (int j = 0; j < second_derivatives[d]->cols(); ++j)
					copy->second_derivatives[d]->at(i, j) = this->second_derivatives[d]->at(i, j);

		return copy;
	}
};

//TODO
/*template<int features, int rows, int cols, int out_rows, int out_cols, int out_features, int connections, int activation_function>
class PerceptronLocalConnectivityLayer : public ILayer
{
public:
	PerceptronLocalConnectivityLayer<features, rows, cols, out_rows, out_cols, out_features, connections, activation_function>()
	{
		activation = activation_function;
		type = CNN_PERCEPTRON_LOCAL_CONNECTIVITY;
		connections_per_neuron = connections;
		feature_maps = std::vector<IMatrix<float>*>(features);
		biases = std::vector<IMatrix<float>*>(out_features);
		recognition_data = std::vector<IMatrix<float>*>(1);
		generative_data = std::vector<IMatrix<float>*>(1);
		second_derivatives = std::vector<IMatrix<float>*>(1);

		for (int k = 0; k < features; ++k)
			feature_maps[k] = new Matrix2D<float, rows, cols>();

		for (int k = 0; k < out_features; ++k)
			biases[k] = new Matrix2D<float, out_rows, out_cols>();

		const float s_d = 1.0f / connections;

		recognition_data[0] = new Matrix2D<float, connections, rows * cols * features>();
		generative_data[0] = new Matrix2D<float, connections, rows * cols * features>();
		second_derivatives[0] = new Matrix2D<float, connections, rows * cols * features>();
		for (int i = 0; i < connections; ++i)
		{
			for (int j = 0; j < rows * cols * features; ++j)
			{
				//uniformally distributed
				float x = (2.0f * s_d * rand()) / RAND_MAX - s_d;
				recognition_data[0]->at(i, j) = x;
				generative_data[0]->at(i, j) = recognition_data[0]->at(i, j);
				second_derivatives[0]->at(i, j) = 0;
			}
		}
	}

	~PerceptronLocalConnectivityLayer<features, rows, cols, out_rows, out_cols, out_features, connections, activation_function>()
	{
		delete recognition_data[0];
		delete generative_data[0];
		delete second_derivatives[0];
		for (int i = 0; i < features; ++i)
			delete feature_maps[i];
		for (int i = 0; i < out_features; ++i)
			delete biases[i];
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		//reset layer
		for (int f_0 = 0; f_0 < out_features; ++f_0)
			for (int i_0 = 0; i_0 < out_rows; ++i_0)
				for (int j_0 = 0; j_0 < out_cols; ++j_0)
					output[f_0]->at(i_0, j_0) = 0;

		const int offset_o = (out_features * out_rows * out_cols - features * rows * cols) / 2;
		int offset = offset_o;

		bool middle = false;

		int n = 0;

		int i_0 = 0;
		int j_0 = 0;
		int f_0 = 0;

		//first half
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					for (int k = 0; k < connections; ++k)
					{
						j_0 = n % out_cols;
						i_0 = ((n - j_0) % (out_rows));
						f_0 = (n - j_0 - i_0 * out_cols) / (out_rows * out_cols);

						//add to weight
						output[f_0]->at(i_0, j_0) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * feature_maps[f]->at(i, j);
						//increment neuron index
						++n;
					}

					if (f * rows * cols + i * cols + j >= features * rows * cols / 2)
					{
						middle = true;
						break;
					}

					//adjust offset, next neuron indices
					if (offset_o >= 0)
					{
						if (offset > 0)
							--offset;
						n -= (connections - 1) / 2 + 1 - offset;
					}

					else
					{
						if (offset < 0)
						{
							++offset;
							n = 0;
						}

						else
							n -= (connections - 1) / 2 + 1;
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		middle = false;

		offset = offset_o;
		n = out_features * out_rows * out_cols - 1;
		i_0 = out_rows - 1;
		j_0 = out_cols - 1;
		f_0 = out_features - 1;


		//second half
		for (int f = features - 1; f >= 0; --f)
		{
			for (int i = rows - 1; i >= 0; --i)
			{
				for (int j = cols - 1; j >= 0; --j)
				{
					if (f * rows * cols + i * cols + j <= features * rows * cols / 2)
					{
						middle = true;
						break;
					}

					for (int k = connections - 1; k >= 0; --k)
					{
						j_0 = n % out_cols;
						i_0 = ((n - j_0) % (out_rows));
						f_0 = (n - j_0 - i_0 * out_cols) / (out_rows * out_cols);

						//add to weight
						output[f_0]->at(i_0, j_0) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * feature_maps[f]->at(i, j);
						//decrement neuron index
						--n;
					}

					//adjust offset, next neuron indices
					if (offset_o >= 0)
					{
						if (offset > 0)
							--offset;
						n += (connections - 1) / 2 + 1 - offset;
					}

					else
					{
						if (offset < 0)
						{
							++offset;
							n = out_features * out_rows * out_cols - 1;
						}

						else
							n += (connections - 1) / 2 + 1;
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		//bias
		if (use_biases)
			for (int f_0 = 0; f_0 < out_features; ++f_0)
				for (int i_0 = 0; i_0 < out_rows; ++i_0)
					for (int j_0 = 0; j_0 < out_cols; ++j_0)
						output[f_0]->at(i_0, j_0) += biases[f_0]->at(i_0, j_0);
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//reset layers
		for (int f = 0; f < features; ++f)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					feature_maps[f]->at(i, j) = 0;

		//Think of as "leftover neuron" count divided by two
		const int offset_o = (out_features * out_rows * out_cols - features * rows * cols) / 2;
		int offset = offset_o;

		bool middle = false;

		int n = 0;
		int f_0 = 0;
		int i_0 = 0;
		int j_0 = 0;

		//first half
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					for (int k = 0; k < connections; ++k)
					{
						j_0 = n % out_cols;
						i_0 = ((n - j_0) % (out_rows));
						f_0 = (n - j_0 - i_0 * out_cols) / (out_rows * out_cols);

						//add to weight
						if (!use_g_weights)
							feature_maps[f]->at(i, j) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);
						else
							feature_maps[f]->at(i, j) += generative_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);

						//increment neuron index
						++n;
					}

					if (f * rows * cols + i * cols + j >= features * rows * cols / 2)
					{
						middle = true;
						break;
					}

					//adjust offset, next neuron indices
					if (offset_o >= 0)
					{
						if (offset > 0)
							--offset;
						n -= (connections - 1) / 2 + 1 - offset;
					}

					else
					{
						if (offset < 0)
						{
							++offset;
							n = 0;
						}

						else
							n -= (connections - 1) / 2 + 1;
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		middle = false;

		offset = offset_o;
		n = out_features * out_rows * out_cols - 1;
		i_0 = out_rows - 1;
		j_0 = out_cols - 1;
		f_0 = out_features - 1;

		//second half
		for (int f = features - 1; f >= 0; --f)
		{
			for (int i = rows - 1; i >= 0; --i)
			{
				for (int j = cols - 1; j >= 0; --j)
				{
					if (f * rows * cols + i * cols + j <= features * rows * cols / 2)
					{
						middle = true;
						break;
					}

					for (int k = connections - 1; k >= 0; --k)
					{
						j_0 = n % out_cols;
						i_0 = ((n - j_0) % (out_rows));
						f_0 = (n - j_0 - i_0 * out_cols) / (out_rows * out_cols);

						//add to weight
						if (!use_g_weights)
							feature_maps[f]->at(i, j) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);
						else
							feature_maps[f]->at(i, j) += generative_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);

						//decrement neuron index
						--n;
					}

					if (f * rows * cols + i * cols + j <= features * rows * cols / 2)
					{
						middle = true;
						break;
					}

					//adjust offset, next neuron indices
					if (offset_o >= 0)
					{
						if (offset > 0)
							--offset;
						n += (connections - 1) / 2 + 1 - offset;
					}

					else
					{
						if (offset < 0)
						{
							++offset;
							n = out_features * out_rows * out_cols - 1;
						}

						else
							n += (connections - 1) / 2 + 1;
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}
	}

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
		//reset layer
		for (int f_0 = 0; f_0 < out_features; ++f_0)
			for (int i_0 = 0; i_0 < out_rows; ++i_0)
				for (int j_0 = 0; j_0 < out_cols; ++j_0)
					output[f_0]->at(i_0, j_0) = 0;

		const int offset_o = (out_features * out_rows * out_cols - features * rows * cols) / 2;
		int offset = offset_o;

		bool middle = false;

		int f_0 = 0;
		int i_0 = 0;
		int j_0 = 0;

		//first half
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					for (int k = 0; k < connections; ++k)
					{
						//add to weight
						output[f_0]->at(i_0, j_0) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * feature_maps[f]->at(i, j);
						//increment neuron index
						++j_0;
						if (j_0 >= out_cols)
						{
							j_0 = 0;
							++i_0;
							if (i_0 >= out_rows)
							{
								i_0 = 0;
								++f_0;
								if (f_0 * out_rows * out_cols + i_0 * out_cols + j_0 > out_features * out_rows * out_cols / 2)
								{
									middle = true;
									break;
								}
							}
						}
					}

					if (middle)
						break;

					//adjust offset, next neuron indices
					offset /= connections;
					j_0 -= offset + (connections - 1) / 2;
					if (j_0 < 0)
					{
						j_0 += out_cols;
						--i_0;
						if (i_0 < 0)
						{
							i_0 += out_rows;
							--f_0;
						}
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		offset = offset_o;
		f_0 = out_features - 1;
		i_0 = out_rows - 1;
		j_0 = out_cols - 1;

		//second half
		for (int f = features - 1; f > 0; --f)
		{
			for (int i = rows; i > 0; --i)
			{
				for (int j = cols; j > 0; --j)
				{
					for (int k = connections - 1; k > 0; --k)
					{
						//add to weight
						output[f_0]->at(i_0, j_0) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * feature_maps[f]->at(i, j);
						//decrement neuron index
						--j_0;
						if (j_0 < 0)
						{
							j_0 = out_cols - 1;
							--i_0;
							if (i_0 < 0)
							{
								i_0 = out_rows - 1;
								--f_0;
								if (f_0 * out_rows * out_cols + i_0 * out_cols + j_0 < out_features * out_rows * out_cols / 2)
								{
									middle = true;
									break;
								}
							}
						}
					}

					if (middle)
						break;

					//adjust offset, next neuron indices
					offset /= connections;
					j_0 += offset - (connections - 1) / 2;
					if (j_0 < 0)
					{
						j_0 += out_cols;
						--i_0;
						if (i_0 < 0)
						{
							i_0 += out_rows;
							--f_0;
						}
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		//sigmoid & bias
		if (use_biases)
			for (f_0 = 0; f_0 < out_features; ++f_0)
				for (i_0 = 0; i_0 < out_rows; ++i_0)
					for (j_0 = 0; j_0 < out_cols; ++j_0)
						output[f_0]->at(i_0, j_0) = 1 / (1 + exp(-output[f_0]->at(i_0, j_0) + biases[f_0]->at(i_0, j_0)));
		else
			for (f_0 = 0; f_0 < out_features; ++f_0)
				for (i_0 = 0; i_0 < out_rows; ++i_0)
					for (j_0 = 0; j_0 < out_cols; ++j_0)
						output[f_0]->at(i_0, j_0) = 1 / (1 + exp(-output[f_0]->at(i_0, j_0)));
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//reset layers
		for (int f = 0; f < features; ++f)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					feature_maps[f]->at(i, j) = 0;

		const int offset_o = (out_features * out_rows * out_cols - features * rows * cols) / 2;
		int offset = offset_o;

		bool middle = false;

		int f_0 = 0;
		int i_0 = 0;
		int j_0 = 0;

		//first half
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					for (int k = 0; k < connections; ++k)
					{
						//add to weight
						if (!use_g_weights)
							feature_maps[f]->at(i, j) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);
						else
							feature_maps[f]->at(i, j) += generative_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);

						//increment neuron index
						++j_0;
						if (j_0 >= out_cols)
						{
							j_0 = 0;
							++i_0;
							if (i_0 >= out_rows)
							{
								i_0 = 0;
								++f_0;
								if (f_0 * out_rows * out_cols + i_0 * out_cols + j_0 > out_features * out_rows * out_cols / 2)
								{
									middle = true;
									break;
								}
							}
						}
					}

					if (middle)
						break;

					//adjust offset, next neuron indices
					offset /= connections;
					j_0 -= offset + (connections - 1) / 2;
					if (j_0 < 0)
					{
						j_0 += out_cols;
						--i_0;
						if (i_0 < 0)
						{
							i_0 += out_rows;
							--f_0;
						}
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		offset = offset_o;
		f_0 = out_features - 1;
		i_0 = out_rows - 1;
		j_0 = out_cols - 1;

		//second half
		for (int f = features - 1; f > 0; --f)
		{
			for (int i = rows; i > 0; --i)
			{
				for (int j = cols; j > 0; --j)
				{
					for (int k = connections - 1; k > 0; --k)
					{
						//add to weight
						if (!use_g_weights)
							feature_maps[f]->at(i, j) += recognition_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);
						else
							feature_maps[f]->at(i, j) += generative_data[0]->at(k, f * rows * cols + i * cols + j) * input[f_0]->at(i_0, j_0);

						//decrement neuron index
						--j_0;
						if (j_0 < 0)
						{
							j_0 = out_cols - 1;
							--i_0;
							if (i_0 < 0)
							{
								i_0 = out_rows - 1;
								--f_0;
								if (f_0 * out_rows * out_cols + i_0 * out_cols + j_0 < out_features * out_rows * out_cols / 2)
								{
									middle = true;
									break;
								}
							}
						}
					}

					if (middle)
						break;

					//adjust offset, next neuron indices
					offset /= connections;
					j_0 += offset - (connections - 1) / 2;
					if (j_0 < 0)
					{
						j_0 += out_cols;
						--i_0;
						if (i_0 < 0)
						{
							i_0 += out_rows;
							--f_0;
						}
					}
				}
				if (middle)
					break;
			}
			if (middle)
				break;
		}

		for (int f = 0; f < features; ++f)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					feature_maps[f]->at(i, j) = 1 / (1 + exp(-feature_maps[f]->at(i, j)));
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
		//TODO
	}

	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new PerceptronLocalConnectivityLayer<features, rows, cols, out_rows, out_cols, out_features, connections, activation_function>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		for (int f = 0; f < biases.size(); ++f)
			for (int i = 0; i < biases[f]->rows(); ++i)
				for (int j = 0; j < biases[f]->cols(); ++j)
					copy->biases[f]->at(i, j) = this->biases[f]->at(i, j);

		for (int d = 0; d < recognition_data.size(); ++d)
			for (int i = 0; i < recognition_data[d]->rows(); ++i)
				for (int j = 0; j < recognition_data[d]->cols(); ++j)
					copy->recognition_data[d]->at(i, j) = this->recognition_data[d]->at(i, j);

		for (int d = 0; d < generative_data.size(); ++d)
			for (int i = 0; i < generative_data[d]->rows(); ++i)
				for (int j = 0; j < generative_data[d]->cols(); ++j)
					copy->generative_data[d]->at(i, j) = this->generative_data[d]->at(i, j);

		for (int d = 0; d < second_derivatives.size(); ++d)
			for (int i = 0; i < second_derivatives[d]->rows(); ++i)
				for (int j = 0; j < second_derivatives[d]->cols(); ++j)
					copy->second_derivatives[d]->at(i, j) = this->second_derivatives[d]->at(i, j);

		return copy;
	}
};*/

template<int features, int rows, int cols, int out_rows, int out_cols>
class MaxpoolLayer : public ILayer
{
public:
	MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		type = CNN_MAXPOOL;
		use_biases = false;
		feature_maps = std::vector<IMatrix<float>*>(features);
		switches = std::vector<IMatrix<std::pair<int, int>>*>(features);
		for (int i = 0; i < features; ++i)
		{
			feature_maps[i] = new Matrix2D<float, rows, cols>();
			switches[i] = new Matrix2D<std::pair<int, int>, out_rows, out_cols>();
		}
		recognition_data = std::vector<IMatrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<IMatrix<float>*>(1);
		generative_data[0] = new Matrix2D<float, 0, 0>();
	}

	~MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < recognition_data.size(); ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		for (int f_0 = 0; f_0 < features; ++f_0)
			for (int i = 0; i < output[f_0]->rows(); ++i)
				for (int j = 0; j < output[f_0]->cols(); ++j)
					output[f_0]->at(i, j) = INT_MIN;

		for (int f = 0; f < features; ++f)
		{
			const int down = rows / out_rows;
			const int across = cols / out_cols;
			Matrix2D<Matrix2D<float, down, across>, out_rows, out_cols> samples;

			//get samples
			for (int i = 0; i < out_rows; ++i)
			{
				for (int j = 0; j < out_cols; ++j)
				{
					//get the current sample
					int maxI = (i + 1) * down;
					int maxJ = (j + 1) * across;
					for (int i2 = i * down; i2 < maxI; ++i2)
					{
						for (int j2 = j * across; j2 < maxJ; ++j2)
						{
							samples.at(i, j).at(maxI - i2 - 1, maxJ - j2 - 1) = feature_maps[f]->at(i2, j2);
						}
					}
				}
			}

			//find maxes
			for (int i = 0; i < out_rows; ++i)
			{
				for (int j = 0; j < out_cols; ++j)
				{
					for (int n = 0; n < samples.at(i, j).rows(); ++n)
					{
						for (int m = 0; m < samples.at(i, j).cols(); ++m)
						{
							if (samples.at(i, j).at(n, m) > output[f]->at(i, j))
							{
								output[f]->at(i, j) = samples.at(i, j).at(n, m);
								switches[f]->at(i, j) = std::make_pair(n, m);
							}
						}
					}
				}
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		std::vector<IMatrix<float>*>();
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
		//not applicable
	}

	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
		//just move the values back to which ones were passed on
		for (int f = 0; f < features; ++f)
		{
			const int down = rows / out_rows;
			const int across = cols / out_cols;

			//search each sample
			for (int i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (int j_0 = 0; j_0 < out_cols; ++j_0)
				{
					std::pair<int, int> coords = switches[f]->at(i_0, j_0);
					for (int i = 0; i < down; ++i)
					{
						for (int j = 0; j < across; ++j)
						{
							if (i == coords.first && j == coords.second)
								feature_maps[f]->at(i_0 * down + i, j_0 * across + j) = deriv[f]->at(i_0, j_0);
							else
								feature_maps[f]->at(i * down, j * across) = 0;
						}
					}
				}
			}
		}
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new MaxpoolLayer<features, rows, cols, out_rows, out_cols>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		return copy;
	}

private:
	std::vector<IMatrix<std::pair<int, int>>*> switches;

};

template<int features, int rows, int cols>
class SoftMaxLayer : public ILayer
{
public:
	SoftMaxLayer<features, rows, cols>()
	{
		type = CNN_SOFTMAX;
		use_biases = false;
		feature_maps = std::vector<IMatrix<float>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, cols>();
		recognition_data = std::vector<IMatrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<IMatrix<float>*>(1);
		generative_data[0] = new Matrix2D<float, 0, 0>();
	}

	~SoftMaxLayer<features, rows, cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < recognition_data.size(); ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		for (int f = 0; f < features; ++f)
		{
			//find total
			float sum = 0.0f;
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					sum += exp(feature_maps[f]->at(i, j));

			//get prob
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					output[f]->at(i, j) = exp(feature_maps[f]->at(i, j)) / sum;
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
	}

	//TODO: FIX
	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					//cycle through all again
					for (int i2 = 0; i2 < rows; ++i2)
					{
						for (int j2 = 0; j2 < cols; ++j2)
						{
							float h_i = data[f]->at(i, j);
							float h_j = data[f]->at(i2, j2);
							feature_maps[f]->at(i, j) += (i2 == i && j2 == j) ? h_i * (1 - h_i) : -h_i * h_j;
						}
					}
				}
			}
		}
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new SoftMaxLayer<features, rows, cols>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		return copy;
	}
};

template<int features, int rows, int cols>
class InputLayer : public ILayer
{
public:
	InputLayer<features, rows, cols>()
	{
		type = CNN_INPUT;
		use_biases = false;
		feature_maps = std::vector<IMatrix<float>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, cols>();
		recognition_data = std::vector<IMatrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<IMatrix<float>*>(1);
		generative_data[0] = new Matrix2D<float, 0, 0>();
	}

	~InputLayer<features, rows, cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < recognition_data.size(); ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		//just output
		for (int f = 0; f < features; ++f)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					output[f]->at(i, j) = feature_maps[f]->at(i, j);
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
	}

	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new InputLayer<features, rows, cols>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		return copy;
	}

};

template<int features, int rows, int cols>
class OutputLayer : public ILayer
{
public:
	OutputLayer<features, rows, cols>()
	{
		type = CNN_OUTPUT;
		use_biases = false;
		feature_maps = std::vector<IMatrix<float>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, cols>();
		recognition_data = std::vector<IMatrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<IMatrix<float>*>(1);
		generative_data[0] = new Matrix2D<float, 0, 0>();
	}

	~OutputLayer<features, rows, cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < recognition_data.size(); ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void wake_sleep(float &learning_rate, bool &use_dropout)
	{
	}

	void back_prop(std::vector<IMatrix<float>*> &data, std::vector<IMatrix<float>*> &deriv, std::vector<IMatrix<float>*> &weight_gradient, std::vector<IMatrix<float>*> &bias_gradient, float learning_rate)
	{
	}

	ILayer* clone()
	{
		//create intial copy
		ILayer* copy = new OutputLayer<features, rows, cols>();

		//copy data over
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					copy->feature_maps[f]->at(i, j) = this->feature_maps[f]->at(i, j);

		return copy;
	}
};

