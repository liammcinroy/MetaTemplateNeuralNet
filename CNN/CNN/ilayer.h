#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_MAXPOOL 3

template<unsigned int rows, unsigned int cols, unsigned int kernel_size> Matrix<int>*
convolve(Matrix<int>* &input, Matrix<float>* &kernel, int &stride)
{
	int N = (kernel_size - 1) / 2;
	Matrix2D<int, rows - (kernel_size - 1), cols - (kernel_size - 1)>* output =
		new Matrix2D<int, rows - (kernel_size - 1), cols - (kernel_size - 1)>();

	for (int i = N; i < rows - N; i += stride)
	{
		for (int j = N; j < cols - N; j += stride)
		{
			int sum = 0;
			for (int n = N; n >= -N; --n)
				for (int m = N; m >= -N; --m)
					sum += input->at(i - n, j - m) * kernel->at(N - n, N - m);
			output->at(i - N, j - N) = sum;
		}
	}
	return output;
}

class  ILayer
{
public:
	ILayer() = default;

	virtual ~ILayer() = default;

	virtual std::vector<Matrix<int>*> feed_forwards() = 0;

	virtual std::vector<Matrix<int>*> feed_backwards(std::vector<Matrix<float>*> &weights) = 0;

	void find_probability()
	{
		for (int f = 0; f < feature_maps.size(); ++f)
		{
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
			{
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
				{
					float prob = 1 / (1 + exp((float)-feature_maps[f]->at(i, j)));
					feature_maps[f]->at(i, j) = (rand() <= prob) ? 1 : 0;
				}
			}
		}
	}

	void dropout()
	{
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					if (rand() >= .5f)
						feature_maps[f]->at(i, j) = 0;
	}

	void wake_sleep(ILayer &above, float &learning_rate)
	{
		//find difference via gibbs sampling
		std::vector<Matrix<int>*> discriminated;
		discriminated = this->feed_forwards();
		std::vector<Matrix<int>*> generated;
		generated = above.feed_backwards(data);
		std::vector<Matrix<int>*> temp_feature;
		temp_feature = feature_maps;
		feature_maps = generated;
		std::vector<Matrix<int>*> reconstructed;
		reconstructed = this->feed_forwards();
		feature_maps = temp_feature;

		//adjust weights
		for (int f = 0; f < feature_maps.size(); ++f)
			for (int i = 0; i < data.size(); ++i)
				for (int j = 0; j < feature_maps[f]->rows(); ++j)
					data[f]->at(i, j) += learning_rate * (discriminated[f]->at(i, 0) - reconstructed[f]->at(i, 0));
	}

	void back_propogate();//TODO: investigate backpropogation

	std::vector<Matrix<int>*> feature_maps;

	std::vector<Matrix<float>*> data;

	int type;
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int data_size, unsigned int out_features,
unsigned int in_features, unsigned int in_rows, unsigned int in_cols> class ConvolutionLayer : public ILayer
{
public:
	ConvolutionLayer<features, rows, cols, data_size, out_features, in_features, in_rows, in_cols>()
	{
		type = CNN_CONVOLUTION;
		feature_maps = std::vector<Matrix<int>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<int, rows, cols>();
		data = std::vector<Matrix<float>*>(out_features);
		for (int k = 0; k < data.size(); ++k)
		{
			data[k] = new Matrix2D<float, data_size, data_size>();
			for (int i = 0; i < data_size; ++i)
				for (int j = 0; j < data_size; ++j)
					data[k]->at(i, j) = rand();
		}
	}

	~ConvolutionLayer<features, rows, cols, data_size, out_features, in_features, in_rows, in_cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < data.size(); ++i)
			delete data[i];
	}

	std::vector<Matrix<int>*> feed_forwards()
	{
		std::vector<Matrix<int>*> output(out_features);
		for (int i = 0; i < out_features; ++i)
			output[i] = convolve<rows, cols, data_size>(feature_maps[0], data[i], stride);
		return output;
	}

	std::vector<Matrix<int>*> feed_backwards(std::vector<Matrix<float>*> &weights)
	{
		std::vector<Matrix<int>*> output(in_features);
		for (int f = 0; f < features; ++f)
		{
			Matrix2D<int, in_rows, in_cols>* current = new Matrix2D<int, in_rows, in_cols>();
			for (int i = 0; i < feature_maps[f]->rows(); ++i)
				for (int j = 0; j < feature_maps[f]->cols(); ++j)
					for (int n = 0; n < weights[f]->rows(); ++n)
						for (int m = 0; m < weights[f]->cols(); ++m)
							current->at(i + n, j + m) += feature_maps[f]->at(i, j) * weights[f]->at(n, m);
			output[f] = current;
		}
		return output;
	}

	int stride = 1;
};

template<unsigned int features, unsigned int rows, unsigned int out_rows, unsigned int in_rows> class FeedForwardLayer : public ILayer
{
public:
	FeedForwardLayer<features, rows, out_rows, in_rows>()
	{
		type = CNN_FEED_FORWARD;
		feature_maps = std::vector<Matrix<int>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<int, rows, 1>();
		data = std::vector<Matrix<float>*>(features);
		for (int k = 0; k < data.size(); ++k)
		{
			data[k] = new Matrix2D<float, out_rows, rows>();
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < rows; ++j)
					data[k]->at(i, j) = rand();
		}
	}

	~FeedForwardLayer<features, rows, out_rows, in_rows>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < data.size(); ++i)
			delete data[i];
	}

	std::vector<Matrix<int>*> feed_forwards()
	{
		std::vector<Matrix<int>*> output(features);
		for (int f = 0; f < features; ++f)
		{
			Matrix2D<int, out_rows, 1>* current = new Matrix2D<int, out_rows, 1>();
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < feature_maps[f]->rows(); ++j)
					current->at(i, 0) += (feature_maps[f]->at(j, 0) * data[f]->at(i, j));
			output[f] = current;
		}
		return output;
	}

	std::vector<Matrix<int>*> feed_backwards(std::vector<Matrix<float>*> &weights)
	{
		std::vector<Matrix<int>*> output(features);
		for (int f = 0; f < features; ++f)
		{
			auto current = new Matrix2D<int, in_rows, 1>();
			for (int i = 0; i < in_rows; ++i)
				for (int j = 0; j < feature_maps[f]->rows(); ++j)
					current->at(i, 0) += weights[f]->at(j, i) * feature_maps[f]->at(j, 0);
			output[f] = current;
		}
		return output;
	}
};

template<unsigned int features, unsigned int rows, unsigned int cols, unsigned int out_rows, unsigned int out_cols> class MaxpoolLayer : public ILayer
{
public:
	MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		type = CNN_MAXPOOL;
		feature_maps = std::vector<Matrix<int>*>(features);
		for (int i = 0; i < features; ++i)
			feature_maps[i] = new Matrix2D<int, rows, cols>();
		data = std::vector<Matrix<float>*>(1);
		data[0] = new Matrix2D<float, 0, 0>();
	}

	~MaxpoolLayer<features, rows, cols, out_rows, out_cols>()
	{
		for (int i = 0; i < feature_maps.size(); ++i)
			delete feature_maps[i];
		for (int i = 0; i < data.size(); ++i)
			delete data[i];
	}

	std::vector<Matrix<int>*> feed_forwards()
	{
		std::vector<Matrix<int>*> output(features);
		for (int f = 0; f < features; ++f)
		{
			const int down = rows / out_rows;
			const int across = cols / out_cols;
			Matrix2D<Matrix2D<int, rows / out_rows, cols / out_cols>, out_rows, out_cols> samples;
			

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
			Matrix2D<int, out_rows, out_cols>* maxes = new Matrix2D<int, out_rows, out_cols>();
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < out_cols; ++j)
					for (int n = 0; n < samples.at(i, j).rows(); ++n)
						for (int m = 0; m < samples.at(i, j).cols(); ++m)
							if (samples.at(i, j).at(n, m) > maxes->at(i, j))
								maxes->at(i, j) = samples.at(i, j).at(i, j);
			output[f] = maxes;
		}
		return output;
	}

	std::vector<Matrix<int>*> feed_backwards(std::vector<Matrix<float>*> &weights)
	{
		return std::vector<Matrix<int>*>();
	}
};