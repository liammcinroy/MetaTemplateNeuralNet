#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_MAXPOOL 3

template<unsigned int rows, unsigned int cols, unsigned int kernel_size> Matrix<float>*
convolve(Matrix<float>* &input, Matrix<float>* &kernel, int &stride)
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
			output->at(i - N, j - N) = sum;
		}
	}
	return output;
}

template<unsigned int rows, unsigned int cols, unsigned int kernel_size> Matrix<float>*
convolve_prob(Matrix<float>* &input, Matrix<float>* &kernel, int &stride)
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

			float prob = 1 / (1 + exp((float)-sum));
			output->at(i - N, j - N) = ((1.0f * rand()) / RAND_MAX <= prob) ? 1 : 0;
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
			for (int f = 0; f < feature_maps.size(); ++f)
			{
				for (int i = 0; i < recognition_data[f]->rows(); ++i)
				{
					for (int j = 0; j < feature_maps[f]->rows(); ++j)
					{
						float delta_w = learning_rate * (discriminated[f]->at(i, 0) - reconstructed[f]->at(i, 0));
						recognition_data[f]->at(i, j) -= delta_w;
						generative_data[f]->at(i, j) += delta_w;
					}
				}
			}
		}

		else
		{
			for (int f = 0; f < feature_maps.size(); ++f)
			{
				int r = (generative_data[f]->rows() - 1) / 2;
				for (int i = 0; i < reconstructed[f]->rows(); ++i)
				{
					for (int j = 0; j < reconstructed[f]->cols(); ++j)
					{
						for (int i2 = r; i2 >= -r; --i2)
						{
							for (int j2 = r; j2 >= -r; --j2)
							{
								int adj_i = i - i2;
								int adj_j = j - j2;

								if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < reconstructed[f]->rows() && adj_j + r < reconstructed[f]->cols())
								{
									for (int n = r; n >= -r; --n)
									{
										for (int m = r; m >= -r; --m)
										{
											if (adj_i - n == i && adj_j - m == j)
											{
												float delta_w = learning_rate *
													(discriminated[f]->at(i, j) - reconstructed[f]->at(i, j));
												recognition_data[f]->at(r - n, r - m) -= delta_w;
												generative_data[f]->at(r - n, r - m) += delta_w;
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
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, cols>();

		recognition_data = std::vector<Matrix<float>*>(out_features);
		generative_data = std::vector<Matrix<float>*>(out_features);
		for (int k = 0; k < recognition_data.size(); ++k)
		{
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
			delete feature_maps[i];
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
			output[i] = convolve<rows, cols, recognition_data_size>(feature_maps[0], recognition_data[i], stride);
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f = 0; f < features; ++f)
		{
			int r = (recognition_data[f]->rows() - 1) / 2;
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
			{
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
				{
					float sum = 0.0f;
					for (int i2 = r; i2 >= -r; --i2)
					{
						for (int j2 = r; j2 >= -r; --j2)
						{
							int adj_i = i - i2;
							int adj_j = j - j2;

							if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < feature_maps[f]->rows() && adj_j + r < feature_maps[f]->cols())
							{
								for (int n = r; n >= -r; --n)
								{
									for (int m = r; m >= -r; --m)
									{
										if (adj_i - n == i && adj_j - m == j)
										{
											if (use_g_weights)
												sum += input[f]->at(r - adj_i, r - adj_j) * generative_data[f]->at(r - n, r - m);
											else
												sum += input[f]->at(r - adj_i, r - adj_j) * recognition_data[f]->at(r - n, r - m);
										}
									}
								}
							}
						}
					}
					feature_maps[f]->at(i, j) = sum;
				}
			}
		}
		return feature_maps;
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		std::vector<Matrix<float>*> output(out_features);
		for (int i = 0; i < out_features; ++i)
			output[i] = convolve_prob<rows, cols, recognition_data_size>(feature_maps[0], recognition_data[i], stride);
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f = 0; f < features; ++f)
		{
			int r = (recognition_data[f]->rows() - 1) / 2;
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
			{
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
				{
					float sum = 0.0f;
					for (int i2 = r; i2 > -r; --i2)
					{
						for (int j2 = r; j2 > -r; --j2)
						{
							int adj_i = i - i2;
							int adj_j = j - j2;

							if (0 <= adj_i - r && 0 <= adj_j - r && adj_i + r < feature_maps[f]->rows() && adj_j + r < feature_maps[f]->cols())
							{
								for (int n = r; r > -r; --n)
								{
									for (int m = r; r > -r; --m)
									{
										if (adj_i - n == i && adj_j - m == j)
										{
											if (use_g_weights)
												sum += input[f]->at(r - adj_i, r - adj_j) * generative_data[f]->at(r - n, r - m);
											else
												sum += input[f]->at(adj_i, adj_j) * recognition_data[f]->at(r - n, r - m);
										}
									}
								}
							}
						}
					}
					float prob = 1 / (1 + exp(-sum));
					feature_maps[f]->at(i, j) = (rand() / RAND_MAX >= prob) ? 1 : 0;
				}
			}
		}
		return feature_maps;
	}

	int stride = 1;
};

template<unsigned int features, unsigned int rows, unsigned int out_rows> 
class FeedForwardLayer : public ILayer
{
public:
	FeedForwardLayer<features, rows, out_rows>()
	{
		type = CNN_FEED_FORWARD;
		feature_maps = std::vector<Matrix<float>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<float, rows, 1>();
		recognition_data = std::vector<Matrix<float>*>(features);
		generative_data = std::vector<Matrix<float>*>(features);
		for (int k = 0; k < recognition_data.size(); ++k)
		{
			recognition_data[k] = new Matrix2D<float, out_rows, rows>();
			generative_data[k] = new Matrix2D<float, out_rows, rows>();
			for (int i = 0; i < out_rows; ++i)
			{
				for (int j = 0; j < rows; ++j)
				{
					recognition_data[k]->at(i, j) = (1.0f * rand()) / RAND_MAX;
					generative_data[k]->at(i, j) = recognition_data[k]->at(i, j);
				}
			}
		}
	}

	~FeedForwardLayer<features, rows, out_rows>()
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
			Matrix2D<float, out_rows, 1>* current = new Matrix2D<float, out_rows, 1>();

			for (int i = 0; i < out_rows; ++i)
			{
				float sum = 0.0f;
				for (int j = 0; j < feature_maps[f]->rows(); ++j)
					sum += (feature_maps[f]->at(j, 0) * recognition_data[f]->at(i, j));
				current->at(i, 0) = sum;
			}
			output[f] = current;
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				float sum = 0.0f;
				for (int j = 0; j < input[f]->rows(); ++j)
				{
					if (use_g_weights)
						sum += generative_data[f]->at(j, i) * input[f]->at(j, 0);
					else
						sum += recognition_data[f]->at(j, i) * input[f]->at(j, 0);
				}
				feature_maps[f]->at(i, 0) = sum;
			}
		}
		return feature_maps;
	}

	std::vector<Matrix<float>*> feed_forwards_prob()
	{
		std::vector<Matrix<float>*> output(features);
		for (int f = 0; f < features; ++f)
		{
			Matrix2D<float, out_rows, 1>* current = new Matrix2D<float, out_rows, 1>();

			for (int i = 0; i < out_rows; ++i)
			{
				float sum = 0.0f;
				for (int j = 0; j < feature_maps[f]->rows(); ++j)
					sum += (feature_maps[f]->at(j, 0) * recognition_data[f]->at(i, j));
				float prob = 1 / (1 + exp((float)-sum));
				current->at(i, 0) = ((1.0f * rand()) / RAND_MAX <= prob) ? 1 : 0;
			}
			output[f] = current;
		}
		return output;
	}

	std::vector<Matrix<float>*> feed_backwards_prob(std::vector<Matrix<float>*> &input, const bool &use_g_weights)
	{
		for (int f = 0; f < features; ++f)
		{
			for (int i = 0; i < rows; ++i)
			{
				float sum = 0.0f;
				for (int j = 0; j < input[f]->rows(); ++j)
				{
					if (use_g_weights)
						sum += generative_data[f]->at(j, i) * input[f]->at(j, 0);
					else
						sum += recognition_data[f]->at(j, i) * input[f]->at(j, 0);
				}
				float prob = 1 / (1 + exp((float)-sum));
				feature_maps[f]->at(i, 0) = ((1.0f * rand()) / RAND_MAX <= prob) ? 1 : 0;
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
			Matrix2D<Matrix2D<float, rows / out_rows, cols / out_cols>, out_rows, out_cols> samples;


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
							samples.at(i, j).at(maxI - i2, maxJ - j2) = feature_maps[f]->at(i2, j2);
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