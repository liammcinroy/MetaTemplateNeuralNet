#pragma once

#include "imatrix.h"

#define CNN_CONVOLUTION 1
#define CNN_FEED_FORWARD 2
#define CNN_MAXPOOL 3

class ILayer
{
public:
	~ILayer();
	virtual Matrix2D<int>* feed_forwards();
	virtual Matrix2D<int>* feed_backwards(Matrix2D<float>* &weights);
	void find_probability();
	void dropout();
	void wake_sleep(ILayer &above, float &learning_rate);
	void back_propogate();//TODO: investigate backpropogation
	int in_rows;
	int in_cols;
	int in_features;
	int out_rows;
	int out_cols;
	int out_features;
	int feature_maps_count;
	int data_count;
	int type;
	Matrix2D<int>* feature_maps;
	Matrix2D<float>* data;
};

class ConvolutionLayer : public ILayer
{
public:
	ConvolutionLayer(int features, int rows, int cols, int kernel_rows, int kernel_cols, int output_features, int input_features,
		int input_rows, int input_cols, int conv_stride = 1);
	~ConvolutionLayer();
	virtual Matrix2D<int>* feed_forwards()
	{
		Matrix2D<int>* output = (Matrix2D<int>*)malloc(data_count * sizeof(Matrix2D<int>(out_rows, out_cols)));
		for (int i = 0; i < data_count; ++i)
		{
			Matrix2D<int> total = convolve(feature_maps[0], data[i]);
			for (int f = 1; f < feature_maps_count; ++f)
				total = total + convolve(feature_maps[f], data[i]);
			output[i] = total;
		}
		return output;
	}
	virtual Matrix2D<int>* feed_backwards(Matrix2D<float>* &weights)
	{
		Matrix2D<int>* output = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(in_rows, 1)));
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<int> current(in_rows, in_cols);
			for (int i = 0; i < feature_maps[f].rows; ++i)
				for (int j = 0; j < feature_maps[f].cols; ++j)
					for (int n = 0; n < weights[f].rows; ++n)
						for (int m = 0; m < weights[f].cols; ++m)
							current.at(i + n, j + m) += feature_maps[f].at(i, j) * weights[f].at(n, m);
			output[f] = current;
		}
		return output;
	}
private:
	int stride;
	Matrix2D<int> convolve(Matrix2D<int> &input, Matrix2D<float> &kernel);
};

class FeedForwardLayer : public ILayer
{
	FeedForwardLayer(int features, int rows, int output_rows, int input_num);
	~FeedForwardLayer();
	virtual Matrix2D<int>* feed_forwards()
	{
		Matrix2D<int>* output = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(out_rows, 1)));
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<int> current(out_rows, 1);
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < feature_maps[f].rows; ++j)
					current.at(i, 0) += (feature_maps[f].at(j, 0) * data[f].at(i, j));
			output[f] = current;
		}
		return output;
	}
	virtual Matrix2D<int>* feed_backwards(Matrix2D<float>* &weights)
	{
		Matrix2D<int>* output = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(in_rows, 1)));
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<int> current(in_rows, 1);
			for (int i = 0; i < in_rows; ++i)
				for (int j = 0; j < feature_maps[f].rows; ++j)
					current.at(i, 0) += weights[f].at(j, i) * feature_maps[f].at(j, 0);
			output[f] = current;
		}
		return output;
	}
};

class MaxpoolLayer : public ILayer
{
	MaxpoolLayer(int features, int rows, int cols, int maxed_rows, int maxed_cols);
	~MaxpoolLayer();
	virtual Matrix2D<int>* feed_forwards()
	{
		Matrix2D<int>* output = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(out_rows, out_cols)));
		for (int f = 0; f < feature_maps_count; ++f)
		{
			int down = feature_maps[f].rows / out_rows;
			int across = feature_maps[f].cols / out_cols;
			Matrix2D<Matrix2D<int>> samples(down, across);
			

			//get samples
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < out_cols; ++j)
					samples.at(i, j) = feature_maps[f].from(i * down, j * across, (i + 1) * down, (j + 1) * across);

			//find maxes
			Matrix2D<int> maxes(out_rows, out_cols);
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < out_cols; ++j)
					for (int n = 0; n < samples.at(i, j).rows; ++n)
						for (int m = 0; m < samples.at(i, j).cols; ++m)
							if (samples.at(i, j).at(n, m) > maxes.at(i, j))
								maxes.at(i, j) = samples.at(i, j).at(i, j);
			output[f] = maxes;
		}
		return output;
	}
};