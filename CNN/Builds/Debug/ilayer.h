#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_MAXPOOL 3

template<unsigned int rows, unsigned int cols, unsigned int kernel_size> Matrix<float>*
convolve(Matrix<float>* &input, Matrix<float>* &biases, Matrix<float>* &kernel, int &stride)
{
	int N = (kernel_size - 1) / 2;
	Matrix2D<float, rows - (kernel_size - 1), cols - (kernel_size - 1)>* output =
		new Matrix2D<float, rows - (kernel_size - 1), cols - (kernel_size - 1)>();

	for (int i = N; i < rows - N; i += stride)
	{
		for (int j = N; j < cols - N; j += stride)
		{
			float sum = 0;
			for (int n = N; n >= -N; --n)
			for (int m = N; m >= -N; --m)
				sum += input->at(i - n, j - m) * kernel->at(N - n, N - m);
			output->at(i - N, j - N) = sum + biases->at(i - N, j - N);
		}
	}
	return output;
}

class  ILayer
{
public:
	ILayer() = default;

	virtual ~ILayer() = default;

	virtual std::vector<Matrix<float>*> feed_forwards() = 0;

	virtual std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights) = 0;

	virtual std::vector<Matrix<float>*> feed_forwards_prob() = 0;

	virtual std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights) = 0;

	void dropout()
	{
		for (int f = 0; f < feature_maps.size(); ++f)
		for (int i = 0; i < feature_maps[f]->rows(); ++i)
		for (int j = 0; j < feature_maps[f]->cols(); ++j)
		if ((1.0f * rand()) / RAND_MAX >= .5f)
			feature_maps[f]->at(i, j) = 0;
	}

	void wake_sleep(float &learning_rate, bool &binary_net)
	{
		//find difference via gibbs sampling
		std::vector<Matrix<float>*> discriminated;
		if (binary_net)
			discriminated = this->feed_forwards_prob();
		else
			discriminated = this->feed_forwards();

		std::vector<Matrix<float>*> generated;
		if (binary_net)
			generated = this->feed_backwards_prob(discriminated, true);
		else
			generated = this->feed_backwards(discriminated, true);

		std::vector<Matrix<float>*> temp_feature;
		temp_feature = feature_maps;
		feature_maps = generated;

		std::vector<Matrix<float>*> reconstructed;
		if (binary_net)
			reconstructed = this->feed_forwards_prob();
		else
			reconstructed = this->feed_forwards();
		feature_maps = temp_feature;

		//adjust weights
		if (type == CNN_FEED_FORWARD)
		{
			for (int f_o = 0; f_o < reconstructed.size(); ++f_o)
			{
				for (int i = 0; i < reconstructed[f_o]->rows(); ++i)
				{
					float delta_weight = learning_rate * (discriminated[f_o]->at(i, 0) - reconstructed[f_o]->at(i, 0));
					for (int f = 0; f < generated.size(); ++f)
					{
						for (int j = 0; j < generated[f]->rows(); ++j)
						{
							recognition_data[0]->at(i + f_o * reconstructed[f_o]->rows(), j + f * generated[f]->rows()) -= delta_weight;
							generative_data[0]->at(i + f_o * reconstructed[f_o]->rows(), j + f * generated[f]->rows()) += delta_weight;
						}
					}
				}
			}
		}

		else
		{
			for (int f_o = 0; f_o < reconstructed.size(); ++f_o)
			{
				for (int f = 0; f < feature_maps.size(); ++f)
				{
					int r = (generative_data[f_o]->rows() - 1) / 2;
					for (int i = 0; i < reconstructed[f_o]->rows(); ++i)
					{
						for (int j = 0; j < reconstructed[f_o]->cols(); ++j)
						{
							for (int i2 = r; i2 >= -r; --i2)
							{
								for (int j2 = r; j2 >= -r; --j2)
								{
									int adj_i = i - i2;
									int adj_j = j - j2;

									if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < reconstructed[f_o]->rows() &&
										adj_j + r < reconstructed[f_o]->cols())
									{
										for (int n = r; n >= -r; --n)
										{
											for (int m = r; m >= -r; --m)
											{
												if (adj_i - n == i && adj_j - m == j)
												{
													float delta_w = learning_rate *
														(discriminated[f_o]->at(i, j) - reconstructed[f_o]->at(i, j));
													recognition_data[f_o]->at(r - n, r - m) -= delta_w;
													generative_data[f_o]->at(r - n, r - m) += delta_w;
												}
											}
										}
									}
								}
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

	std::vector<Matrix<float>*> feature_maps;

	std::vector<Matrix<float>*> biases;

	std::vector<Matrix<float>*> recognition_data;

	std::vector<Matrix<float>*> generative_data;

	std::vector<Matrix<std::pair<int, int>>*> coords_of_max;

	int type;
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int recognition_data_size, unsigned int out_features>
class ConvolutionLayer : public ILayer
{
public:
	ConvolutionLayer<features, rows, cols, recognition_data_size, out_features>()
	{
		type = CNN_CONVOLUTION;
		feature_maps = std::vector<Matrix<float>*>(features);
		for (int k = 0; k < features; ++k)
		{
			feature_maps[k] = new Matrix2D<float, rows, cols>();
		}

		biases = std::vector<Matrix<float>*>(out_features);
		recognition_data = std::vector<Matrix<float>*>(out_features);
		generative_data = std::vector<Matrix<float>*>(out_features);
		for (int k = 0; k < recognition_data.size(); ++k)
		{
			biases[k] = new Matrix2D<float, rows - (recognition_data_size - 1), cols - (recognition_data_size - 1)>();

			recognition_data[k] = new Matrix2D<float, recognition_data_size, recognition_data_size>();
			generative_data[k] = new Matrix2D<float, recognition_data_size, recognition_data_size>();
			for (int i = 0; i < recognition_data_size; ++i)
			{
				for (int j = 0; j < recognition_data_size; ++j)
				{
					recognition_data[k]->at(i, j) = (1.0f * rand()) / RAND_MAX;
					generative_data[k]->at(i, j) = recognition_data[k]->at(i, j);
				}
			}
		}
	}

	~ConvolutionLayer<features, rows, cols, recognition_data_size, out_features>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
		{
			delete feature_maps[i];
			delete biases[i];
		}
		for (int i = 0; i < recognition_data.size(); ++i)
		{
			delete recognition_data[i];
			delete generative_data[i];
		}
	}

	std::vector<Matrix<float>*> feed_forwards()
	{
		std::vector<Matrix<float>*> output(out_features);
		for (int i = 0; i < out_features; ++i)
		{
			output[i] = new Matrix2D<float, rows + 1 - recognition_data_size, cols + 1 - recognition_data_size>();
			for (int j = 0; j < features; ++j)
				output[i] = add<float, rows + 1 - recognition_data_size, cols + 1 - recognition_data_size>(output[i], 
				convolve<rows, cols, recognition_data_size>(feature_maps[j], biases[i], recognition_data[i], stride));
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		//Do the first only
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			int r = (recognition_data[f_o]->rows() - 1) / 2;
			for (int i = 0; i < feature_maps[0]->rows(); ++i)
			{
				for (int j = 0; j < feature_maps[0]->cols(); ++j)
				{
					float sum = 0.0f;
					for (int i2 = r; i2 >= -r; --i2)
					{
						for (int j2 = r; j2 >= -r; --j2)
						{
							int adj_i = i - i2;
							int adj_j = j - j2;

							if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < feature_maps[0]->rows() && adj_j + r < feature_maps[0]->cols())
							{
								for (int n = r; n >= -r; --n)
								{
									for (int m = r; m >= -r; --m)
									{
										if (adj_i - n == i && adj_j - m == j)
										{
											if (use_g_weights)
												sum += input[f_o]->at(adj_i - r, adj_j - r) * generative_data[f_o]->at(r - n, r - m);
											else
												sum += input[f_o]->at(adj_i - r, adj_j - r) * recognition_data[f_o]->at(r - n, r - m);
										}
									}
								}
							}
						}
					}
					feature_maps[0]->at(i, j) = sum;
				}
			}
		}

		//copy as they are congruent
#pragma warning(suppress: 6294)
		for (int f = 1; f < features; ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					feature_maps[f]->at(i, j) = feature_maps[0]->at(i, j);

		return feature_maps;
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		std::vector<Matrix<float>*> output(out_features);
		for (int f = 0; f < out_features; ++f)
		{
			output[f] = new Matrix2D<float, rows + 1 - recognition_data_size, cols + 1 - recognition_data_size>();
			for (int j = 0; j < features; ++j)
				output[f] = add<float, rows + 1 - recognition_data_size, cols + 1 - recognition_data_size>(output[f],
				convolve<rows, cols, recognition_data_size>(feature_maps[j], biases[f], recognition_data[f], stride));
			for (int i = 0; i < rows + 1 - recognition_data_size; ++i)
				for (int j = 0; j < cols + 1 - recognition_data_size; ++j)
					output[f]->at(i, j) = 1 / (1 + exp((float)-output[f]->at(i, j)));
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		//Do the first only
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			int r = (recognition_data[f_o]->rows() - 1) / 2;
			for (int i = 0; i < feature_maps[0]->rows(); ++i)
			{
				for (int j = 0; j < feature_maps[0]->cols(); ++j)
				{
					float sum = 0.0f;
					for (int i2 = r; i2 >= -r; --i2)
					{
						for (int j2 = r; j2 >= -r; --j2)
						{
							int adj_i = i - i2;
							int adj_j = j - j2;

							if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < feature_maps[0]->rows() && adj_j + r < feature_maps[0]->cols())
							{
								for (int n = r; n >= -r; --n)
								{
									for (int m = r; m >= -r; --m)
									{
										if (adj_i - n == i && adj_j - m == j)
										{
											if (use_g_weights)
												sum += input[f_o]->at(adj_i - r, adj_j - r) * generative_data[f_o]->at(r - n, r - m);
											else
												sum += input[f_o]->at(adj_i - r, adj_j - r) * recognition_data[f_o]->at(r - n, r - m);
										}
									}
								}
							}
						}
					}
					feature_maps[0]->at(i, j) = 1 / (1 + exp(-sum));
				}
			}
		}

		//copy as they are congruent
#pragma warning(suppress: 6294)
		for (int f = 1; f < features; ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					feature_maps[f]->at(i, j) = feature_maps[0]->at(i, j);

		return feature_maps;
	}
	int stride = 1;
};

template<unsigned int features, unsigned int rows, unsigned int out_rows, unsigned int out_features>
class FeedForwardLayer : public ILayer
{
public:
	FeedForwardLayer<features, rows, out_rows, out_features>()
	{
		type = CNN_FEED_FORWARD;
		feature_maps = std::vector<Matrix<float>*>(features);
		biases = std::vector<Matrix<float>*>(out_features);
		recognition_data = std::vector<Matrix<float>*>(1);
		generative_data = std::vector<Matrix<float>*>(1);

		for (int k = 0; k < features; ++k)
			feature_maps[k] = new Matrix2D<float, rows, 1>();

		for (int k = 0; k < out_features; ++k)
			biases[k] = new Matrix2D<float, out_rows, 1>();

		recognition_data[0] = new Matrix2D<float, out_rows * out_features, rows * features>();
		generative_data[0] = new Matrix2D<float, out_rows * out_features, rows * features>();
		for (int i = 0; i < out_rows * out_features; ++i)
		{
			for (int j = 0; j < rows * features; ++j)
			{
				recognition_data[0]->at(i, j) = (1.0f * rand()) / RAND_MAX;
				generative_data[0]->at(i, j) = recognition_data[0]->at(i, j);
			}
		}
	}

	~FeedForwardLayer<features, rows, out_rows, out_features>()
	{
		delete recognition_data[0];
		delete generative_data[0];
		for (int i = 0; i < feature_maps.size(); ++i)
		{
			delete feature_maps[i];
			delete biases[i];
		}
	}

	std::vector<Matrix<float>*> feed_forwards()
	{
		std::vector<Matrix<float>*> output(out_features);
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < out_rows; ++i)
				{
					output[f_o] = new Matrix2D<float, out_rows, 1>();
					float sum = 0.0f;
					int row = f_o * out_rows;
					int col = f * rows;
					for (int j = 0; j < rows; ++j)
						sum += (feature_maps[f]->at(j, 0) * recognition_data[0]->at(i + row, j + col));
					output[f_o]->at(i, 0) = sum + biases[f_o]->at(i, 0);
				}
			}
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < rows; ++i)
				{
					float sum = 0.0f;
					for (int j = 0; j < input[f_o]->rows(); ++j)
					{
						int row = f_o * out_rows;
						int col = f * rows;
						if (use_g_weights)
							sum += generative_data[0]->at(j + row, i + col) * input[f_o]->at(j, 0);
						else
							sum += recognition_data[0]->at(j + row, i + col) * input[f_o]->at(j, 0);
					}
					feature_maps[f]->at(i, 0) = sum;
				}
			}
		}
		return feature_maps;
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		std::vector<Matrix<float>*> output(out_features);
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < out_rows; ++i)
				{
					output[f_o] = new Matrix2D<float, out_rows, 1>();
					float sum = 0.0f;
					int row = f_o * out_rows;
					int col = f * rows;
					for (int j = 0; j < rows; ++j)
						sum += (feature_maps[f]->at(j, 0) * recognition_data[0]->at(i + row, j + col));
					output[f_o]->at(i, 0) = 1 / (1 + exp(-(float)(sum + biases[f_o]->at(i, 0))));
				}
			}
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f_o = 0; f_o < out_features; ++f_o)
		{
			for (int f = 0; f < features; ++f)
			{
				for (int i = 0; i < rows; ++i)
				{
					float sum = 0.0f;
					int row = f_o * out_rows;
					int col = f * rows;
					for (int j = 0; j < input[f_o]->rows(); ++j)
					{
						if (use_g_weights)
							sum += generative_data[0]->at(j + row, i + col) * input[f_o]->at(j, 0);
						else
							sum += recognition_data[0]->at(j + row, i + col) * input[f_o]->at(j, 0);
					}
					feature_maps[f]->at(i, 0) = 1 / (1 + exp(-(float)sum));
				}
			}
		}
		return feature_maps;
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int out_rows, unsigned int out_cols>
class MaxpoolLayer : public ILayer
{
public:
	MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		type = CNN_MAXPOOL;
		feature_maps = std::vector<Matrix<float>*>(features);
		coords_of_max = std::vector<Matrix<std::pair<int, int>>*>(features);
		for (int i = 0; i < features; ++i)
		{
			feature_maps[i] = new Matrix2D<float, rows, cols>();
			coords_of_max[i] = new Matrix2D<std::pair<int, int>, out_rows, out_cols>();
		}
		recognition_data = std::vector<Matrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<Matrix<float>*>(1);
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

	std::vector<Matrix<float>*> feed_forwards()
	{
		std::vector<Matrix<float>*> output(features);
		for (int f = 0; f < features; ++f)
		{
			output[f] = new Matrix2D<float, out_rows, out_cols>();
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
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<Matrix<float>*>();
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		std::vector<Matrix<float>*> output(features);
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
			Matrix2D<float, out_rows, out_cols>* maxes = new Matrix2D<float, out_rows, out_cols>();
			for (int i = 0; i < out_rows; ++i)
			{
				for (int j = 0; j < out_cols; ++j)
				{
					for (int n = 0; n < samples.at(i, j).rows(); ++n)
					{
						for (int m = 0; m < samples.at(i, j).cols(); ++m)
						{
							if (samples.at(i, j).at(n, m) > maxes->at(i, j))
							{
								maxes->at(i, j) = 1;//must be one in binary
								coords_of_max[f]->at(i, j) = std::make_pair<int, int>(samples.at(i, j).rows() * i + n, samples.at(i, j).cols() * j + m);
								break;
							}
						}
					}
				}
			}
			output[f] = maxes;
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<Matrix<float>*>();
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols>
class OutputLayer : public ILayer
{
public:
	OutputLayer<features, rows, cols>()
	{
		type = CNN_MAXPOOL;
		feature_maps = std::vector<Matrix<float>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, cols>();
		recognition_data = std::vector<Matrix<float>*>(1);
		recognition_data[0] = new Matrix2D<float, 0, 0>();
		generative_data = std::vector<Matrix<float>*>(1);
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

	std::vector<Matrix<float>*> feed_forwards()
	{
		return std::vector<Matrix<float>*>();
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<Matrix<float>*>();
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		return std::vector<Matrix<float>*>();
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		return std::vector<Matrix<float>*>();
	}
};