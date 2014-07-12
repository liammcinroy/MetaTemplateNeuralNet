#pragma once

#include <vector>

template<class T> class matrix
{
public:
	matrix<T>()
	{
	}
	matrix<T>(int width, int height, int depth)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>();
		for (int k = 0; k < depth; ++k)
		{
			m_cells.push_back(std::vector<std::vector<T>>());
			for (int i = 0; i < width; ++i)
			{
				m_cells[k].push_back(std::vector<T>());
				for (int j = 0; j < height; ++j)
				{
					m_cells[k][i].push_back(T());
				}
			}
		}
		cols = width;
		rows = height;
		dims = depth;
	}
	matrix<T>(int width, int height, int depth, T defaultValue)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>();
		for (int k = 0; k < depth; ++k)
		{
			m_cells.push_back(std::vector<std::vector<T>>());
			for (int i = 0; i < width; ++i)
			{
				m_cells[k].push_back(std::vector<T>());
				for (int j = 0; j < height; ++j)
				{
					m_cells[k][i].push_back(defaultValue);
				}
			}
		}
		cols = width;
		rows = height;
		dims = depth;
	}
	~matrix<T>()
	{
	}
	int rows;
	int cols;
	int dims;
	T at(int i, int j, int k)
	{
		return m_cells[k][j][i];
	}
	void set(int i, int j, int k, T value)
	{
		m_cells[k][j][i] = value;
	}
	matrix<T> at_channel(int k)
	{
		matrix result;
		result = m_cells[k];
		return result;
	}
	matrix<T> from(int left, int top, int width, int height)
	{
		matrix sample(width, height, dims);

		for (int k = 0; k < dims; ++k)
		for (int i = top; i < left + width; ++i)
		for (int j = left; j < top + height; ++j)
			sample.set(i, j, k, (*this).at(i, j, k));
		return sample;
	}
	matrix<T> operator+=(matrix other)
	{
		for (int i = 0; i < other.cols; ++i)
		for (int j = 0; j < other.rows; ++j)
			(*this).set(i, j, 0, (*this).at(i, j, 0) + other.at(i, j, 0));
		return *this;
	}
	matrix<T> operator=(std::vector<std::vector<std::vector<T>>> arr)
	{
		m_cells = arr;
		dims = arr.size();
		cols = arr[0].size();
		rows = arr[0][0].size();
		return *this;
	}
	matrix<T> operator=(std::vector<std::vector<T>> arr)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>(1);
		m_cells[0] = arr;
		dims = 1;
		cols = arr.size();
		rows = arr[0].size();
		return *this;
	}
private:
	std::vector<std::vector<std::vector<T>>> m_cells;
};

