#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_CONVOLUTION 1
#define CNN_PERCEPTRON 2
#define CNN_MAXPOOL 3
#define CNN_SOFTMAX 4
#define CNN_OUTPUT 5

template<unsigned int rows, unsigned int cols, unsigned int kernel_size, unsigned int stride> IMatrix<float>*
convolve(IMatrix<float>* &input, IMatrix<float>* &kernel)
{
	int N = (kernel_size - 1) / 2;
	Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>* output =
		new Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>();

	for (int i = N; i < (rows - kernel_size) + stride; i += stride)
	{
		for (int j = N; j < (cols - kernel_size) + stride; j += stride)
		{
			float sum = 0;
			for (int n = N; n >= -N; --n)
				for (int m = N; m >= -N; --m)
					sum += input->at(i - n, j - m) * kernel->at(N - n, N - m);
			output->at(i / stride, j / stride) = sum;
		}
	}
	return output;
}

template<unsigned int rows, unsigned int cols, unsigned int kernel_size, unsigned int stride> IMatrix<float>*
convolve_back(IMatrix<float>* &input, IMatrix<float>* &kernel)
{
	int N = (kernel_size - 1) / 2;
	Matrix2D<float, rows, cols>* output = new Matrix2D<float, rows, cols>();

	int times_across = 0;
	int times_down = 0;

	for (int i = N; i < (rows - kernel_size) + stride; i += stride)
	{
		for (int j = N; j < (cols - kernel_size) + stride; j += stride)
		{
			float sum = 0;
			for (int n = N; n >= -N; --n)
			{
				for (int m = N; m >= -N; --m)
				{
					output->at(i - N, j - N) += kernel->at(N - n, N - m) * input->at(times_down, times_across);
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

	virtual void feed_forwards_prob(std::vector<IMatrix<float>*> &output) = 0;

	virtual void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights) = 0;

	virtual void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout) = 0;

	std::vector<IMatrix<float>*> feature_maps;

	std::vector<IMatrix<float>*> biases;

	std::vector<IMatrix<float>*> recognition_data;

	std::vector<IMatrix<float>*> generative_data;

	std::vector<IMatrix<std::pair<int, int>>*> coords_of_max;

	int type;

private:
	void dropout(std::vector<IMatrix<float>*> feature_maps)
	{
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					if ((1.0f * rand()) / RAND_MAX >= .5f)
						feature_maps[f]->at(i, j) = 0;
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols,
	unsigned int kernel_size, unsigned int stride, unsigned int out_features>
class ConvolutionLayer : public ILayer
{
public:
	ConvolutionLayer<features, rows, cols, kernel_size, stride, out_features>()
	{
		type = CNN_CONVOLUTION;
		feature_maps = std::vector<IMatrix<float>*>(features);
		for (int k = 0; k < features; ++k)
		{
			feature_maps[k] = new Matrix2D<float, rows, cols>();
		}

		recognition_data = std::vector<IMatrix<float>*>(out_features);
		generative_data = std::vector<IMatrix<float>*>(out_features);
		for (int k = 0; k < recognition_data.size(); ++k)
		{
			recognition_data[k] = new Matrix2D<float, kernel_size, kernel_size>();
			generative_data[k] = new Matrix2D<float, kernel_size, kernel_size>();
			for (int i = 0; i < kernel_size; ++i)
			{
				for (int j = 0; j < kernel_size; ++j)
				{
					recognition_data[k]->at(i, j) = (1.0f * rand()) / RAND_MAX;
					generative_data[k]->at(i, j) = recognition_data[k]->at(i, j);
				}
			}
		}
	}

	~ConvolutionLayer<features, rows, cols, kernel_size, stride, out_features>()
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
		for (int f = 0; f < out_features; ++f)
		{
			for (int j = 0; j < features; ++j)
			{
				add<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>
					(output[f], convolve<rows, cols, kernel_size, stride>(feature_maps[j], recognition_data[f]));
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//Do the first only
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			if (!use_g_weights)
				add<float, rows, cols>(feature_maps[0],
				convolve_back<rows, cols, kernel_size, stride>(input[0], recognition_data[f_o]));
			else
				add<float, rows, cols>(feature_maps[0],
				convolve_back<rows, cols, kernel_size, stride>(input[0], generative_data[f_o]));
		}

		//copy as they are congruent
#pragma warning(suppress: 6294)
		for (int f = 1; f < features; ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					feature_maps[f]->at(i, j) = feature_maps[0]->at(i, j);
	}

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
		for (int f = 0; f < out_features; ++f)
		{
			for (int j = 0; j < features; ++j)
			{
				add<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>
					(output[f], convolve<rows, cols, kernel_size, stride>(feature_maps[j], recognition_data[f]));
			}
			for (int i = 0; i < (rows - kernel_size) / stride + 1; ++i)
				for (int j = 0; j < (cols - kernel_size) / stride + 1; ++j)
					output[f]->at(i, j) = 1 / (1 + exp((float)-output[f]->at(i, j)));
		}
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		//Do the first only
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			if (!use_g_weights)
				add<float, rows, cols>(feature_maps[0],
				convolve_back<rows, cols, kernel_size, stride>(input[0], recognition_data[f_o]));
			else
				add<float, rows, cols>(feature_maps[0],
				convolve_back<rows, cols, kernel_size, stride>(input[0], generative_data[f_o]));
		}

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				feature_maps[0]->at(i, j) = 1 / (1 + exp(-feature_maps[0]->at(i, j)));

		//copy as they are congruent
#pragma warning(suppress: 6294)
		for (int f = 1; f < features; ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					feature_maps[f]->at(i, j) = feature_maps[0]->at(i, j);
	}

	void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout)
	{
		//find difference via gibbs sampling
		std::vector<IMatrix<float>*> discriminated(out_features);
		for (int i = 0; i < out_features; ++i)
			discriminated[i] = new Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>();

		if (binary_net)
			this->feed_forwards_prob(discriminated);
		else
			this->feed_forwards(discriminated);

		if (binary_net)
			this->feed_backwards_prob(discriminated, true);
		else
			this->feed_backwards(discriminated, true);

		std::vector<IMatrix<float>*> reconstructed(out_features);
		for (int i = 0; i < out_features; ++i)
			reconstructed[i] = new Matrix2D<float, (rows - kernel_size) / stride + 1, (cols - kernel_size) / stride + 1>();

		if (binary_net)
			this->feed_forwards_prob(reconstructed);
		else
			this->feed_forwards(reconstructed);

		//adjust weights
		for (int f_o = 0; f_o < reconstructed.size(); ++f_o)
		{
			int N = (generative_data[f_o]->rows() - 1) / 2;
			int times_down = 0;
			int times_across = 0;

			for (int i = N; i < (rows - generative_data[f_o]->rows()) + stride; i += stride)
			{
				for (int j = N; j < (cols - generative_data[f_o]->rows()) + stride; j += stride)
				{
					float delta_w = -learning_rate *
						(discriminated[f_o]->at(times_across, times_down) - reconstructed[f_o]->at(times_across, times_down));

					for (int n = N; n >= -N; --n)
					{
						for (int m = N; m >= -N; --m)
						{
							recognition_data[f_o]->at(N - n, N - m) += delta_w;
							generative_data[f_o]->at(N - n, N - m) -= delta_w;
						}
					}
					++times_across;
				}
				times_across = 0;
				++times_down;
			}
		}

		for (int i = 0; i < reconstructed.size(); ++i)
			delete reconstructed[i];
		for (int i = 0; i < discriminated.size(); ++i)
			delete discriminated[i];
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int out_rows, unsigned int out_cols, unsigned int out_features>
class PerceptronLayer : public ILayer
{
public:
	PerceptronLayer<features, rows, cols, out_rows, out_cols, out_features>()
	{
		type = CNN_PERCEPTRON;
		feature_maps = std::vector<IMatrix<float>*>(features);
		biases = std::vector<IMatrix<float>*>(out_features);
		recognition_data = std::vector<IMatrix<float>*>(1);
		generative_data = std::vector<IMatrix<float>*>(1);

		for (int k = 0; k < features; ++k)
			feature_maps[k] = new Matrix2D<float, rows, cols>();

		for (int k = 0; k < out_features; ++k)
			biases[k] = new Matrix2D<float, out_rows, out_cols>();

		recognition_data[0] = new Matrix2D<float, out_rows * out_cols * out_features, rows * cols * features>();
		generative_data[0] = new Matrix2D<float, out_rows * out_cols * out_features, rows * cols * features>();
		for (int i = 0; i < out_rows * out_cols * out_features; ++i)
		{
			for (int j = 0; j < rows * cols * features; ++j)
			{
				recognition_data[0]->at(i, j) = (1.0f * rand()) / RAND_MAX;
				generative_data[0]->at(i, j) = recognition_data[0]->at(i, j);
			}
		}
	}

	~PerceptronLayer<features, rows, cols, out_rows, out_cols, out_features>()
	{
		delete recognition_data[0];
		delete generative_data[0];
		for (int i = 0; i < features; ++i)
			delete feature_maps[i];
		for (int i = 0; i < out_features; ++i)
			delete biases[i];
	}

	void feed_forwards(std::vector<IMatrix<float>*> &output)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				int row = f_o * out_rows * out_cols;
				int col = f * rows * cols;

				for (int i = 0; i < out_rows; ++i)
				{
					for (int j = 0; j < out_cols; ++j)
					{
						float sum = 0.0f;
						for (int i2 = 0; i2 < rows; ++i2)
							for (int j2 = 0; j2 < cols; ++j2)
								sum += (feature_maps[f]->at(i2, j2) *
								recognition_data[0]->at(f_o * out_rows * out_cols + i * out_cols + j, f * rows * cols + i2 * cols + j2));
						output[f_o]->at(i, j) = sum + biases[f_o]->at(i, j);
					}
				}
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < rows; ++i)
				{
					for (int j = 0; j < cols; ++j)
					{
						float sum = 0.0f;
						for (int i2 = 0; i2 < out_rows; ++i2)
						{
							for (int j2 = 0; j2 < out_cols; ++j2)
							{
								if (use_g_weights)
									sum += generative_data[0]->at(f_o * out_rows * out_cols + i2 * out_cols + j2, f * rows * cols + i * cols + j)
									* input[f_o]->at(i2, j2);
								else
									sum += recognition_data[0]->at(f_o * out_rows * out_cols + i2 * out_cols + j2, f * rows * cols + i * cols + j)
									* input[f_o]->at(i2, j2);
							}
						}
						feature_maps[f]->at(i, j) = sum;
					}
				}
			}
		}
	}

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < out_rows; ++i)
				{
					for (int j = 0; j < out_cols; ++j)
					{
						float sum = 0.0f;
						for (int i2 = 0; i2 < rows; ++i2)
							for (int j2 = 0; j2 < cols; ++j2)
								sum += (feature_maps[f]->at(i2, j2) *
								recognition_data[0]->at(f_o * out_rows * out_cols + i * out_cols + j, f * rows * cols + i2 * cols + j2));
						output[f_o]->at(i, j) = 1 / (1 + exp(-sum + biases[f_o]->at(i, j)));
					}
				}
			}
		}
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				int row = f_o * out_rows * out_cols;
				int col = f * rows * cols;

				for (int i = 0; i < rows; ++i)
				{
					for (int j = 0; j < cols; ++j)
					{
						float sum = 0.0f;
						for (int i2 = 0; i2 < out_rows; ++i2)
						{
							for (int j2 = 0; j2 < out_cols; ++j2)
							{
								if (use_g_weights)
									sum += generative_data[0]->at(f_o * out_rows * out_cols + i2 * out_cols + j2, f * rows * cols + i * cols + j)
									* input[f_o]->at(i2, j2);
								else
									sum += recognition_data[0]->at(f_o * out_rows * out_cols + i2 * out_cols + j2, f * rows * cols + i * cols + j)
									* input[f_o]->at(i2, j2);
							}
						}
						feature_maps[f]->at(i, j) = 1 / (1 + exp(-sum));
					}
				}
			}
		}
	}

	void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout)
	{
		//find difference via gibbs sampling
		std::vector<IMatrix<float>*> discriminated(out_features);
		for (int i = 0; i < out_features; ++i)
			discriminated[i] = new Matrix2D<float, out_rows, out_cols>();

		if (binary_net)
			this->feed_forwards_prob(discriminated);
		else
			this->feed_forwards(discriminated);

		std::vector<IMatrix<float>*> temp_feature(features);
		for (int i = 0; i < features; ++i)
			temp_feature[i] = feature_maps[i]->clone();

		if (binary_net)
			this->feed_backwards_prob(discriminated, true);
		else
			this->feed_backwards(discriminated, true);

		std::vector<IMatrix<float>*> reconstructed(out_features);
		for (int i = 0; i < out_features; ++i)
			reconstructed[i] = new Matrix2D<float, out_rows, out_cols>();

		if (binary_net)
			this->feed_forwards_prob(reconstructed);
		else
			this->feed_forwards(reconstructed);

		for (int i = 0; i < features; ++i)
		{
			*feature_maps[i] = *temp_feature[i];
			delete temp_feature[i];
		}

		//adjust weights
		for (int f_o = 0; f_o < reconstructed.size(); ++f_o)
		{
			for (int i = 0; i < reconstructed[f_o]->rows(); ++i)
			{
				for (int j = 0; j < reconstructed[f_o]->cols(); ++j)
				{
					float delta_weight = -learning_rate * (discriminated[f_o]->at(i, 0) - reconstructed[f_o]->at(i, 0));
					for (int f = 0; f < feature_maps.size(); ++f)
					{
						for (int i2 = 0; i2 < feature_maps[f]->rows(); ++i2)
						{
							for (int j2 = 0; j2 < feature_maps[f]->cols(); ++j2)
							{
								recognition_data[0]->at(f_o * out_rows * out_cols + i * out_cols + j, f * rows * cols + i2 * cols + j2) += delta_weight;
								generative_data[0]->at(f_o * out_rows * out_cols + i * out_cols + j, f * rows * cols + i2 * cols + j2) -= delta_weight;
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
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int out_rows, unsigned int out_cols>
class MaxpoolLayer : public ILayer
{
public:
	MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		type = CNN_MAXPOOL;
		feature_maps = std::vector<IMatrix<float>*>(features);
		coords_of_max = std::vector<IMatrix<std::pair<int, int>>*>(features);
		for (int i = 0; i < features; ++i)
		{
			feature_maps[i] = new Matrix2D<float, rows, cols>();
			coords_of_max[i] = new Matrix2D<std::pair<int, int>, out_rows, out_cols>();
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
								coords_of_max[f]->at(i, j) = std::make_pair<int, int>(samples.at(i, j).rows() * i + n, samples.at(i, j).cols() * j + m);
							}
						}
					}
				}
			}
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<IMatrix<float>*>();
	}

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
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
								coords_of_max[f]->at(i, j) = std::make_pair<int, int>(samples.at(i, j).rows() * i + n, samples.at(i, j).cols() * j + m);
							}
						}
					}
				}
			}
		}
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<IMatrix<float>*>();
	}

	void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout)
	{
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols>
class SoftMaxLayer : public ILayer
{
public:
	SoftMaxLayer<features, rows, cols>()
	{
		type = CNN_SOFTMAX;
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
			float sum = 0.0f;
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					sum += exp(feature_maps[f]->at(i, j));

			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					output[f]->at(i, j) = exp(feature_maps[f]->at(i, j)) / sum;
		}
	}

	void feed_backwards(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
		for (int f = 0; f < features; ++f)
		{
			float sum = 0.0f;
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					sum += exp(feature_maps[f]->at(i, j));

			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					output[f]->at(i, j) = exp(feature_maps[f]->at(i, j)) / sum;
		}
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout)
	{
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols>
class OutputLayer : public ILayer
{
public:
	OutputLayer<features, rows, cols>()
	{
		type = CNN_OUTPUT;
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

	void feed_forwards_prob(std::vector<IMatrix<float>*> &output)
	{
	}

	void feed_backwards_prob(std::vector<IMatrix<float>*> &input, const bool &use_g_weights)
	{
	}

	void wake_sleep(float &learning_rate, bool &binary_net, bool &use_dropout)
	{
	}
};

