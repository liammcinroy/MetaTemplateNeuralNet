#pragma once

#include <vector>
#include <string>
#include <sstream>

template<class T> class matrix
{
public:
	matrix<T>()
	{
	}
	matrix<T>(unsigned int width, unsigned int height, unsigned int depth)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>();
		for (int k = 0; k < depth; ++k)
		{
			m_cells.push_back(std::vector<std::vector<T>>());
			for (int i = 0; i < height; ++i)
			{
				m_cells[k].push_back(std::vector<T>());
				for (int j = 0; j < width; ++j)
				{
					m_cells[k][i].push_back(T());
				}
			}
		}
		cols = width;
		rows = height;
		dims = depth;
	}
	matrix<T>(unsigned int width, unsigned int height, unsigned int depth, T defaultValue)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>();
		for (int k = 0; k < depth; ++k)
		{
			m_cells.push_back(std::vector<std::vector<T>>());
			for (int i = 0; i < height; ++i)
			{
				m_cells[k].push_back(std::vector<T>());
				for (int j = 0; j < width; ++j)
				{
					m_cells[k][i].push_back(defaultValue);
				}
			}
		}
		cols = width;
		rows = height;
		dims = depth;
	}
	matrix<T>(std::vector<std::vector<std::vector<T>>> arr)
	{
		m_cells = arr;
		dims = arr.size();
		rows = arr[0].size();
		cols = arr[0][0].size();
	}
	matrix<T>(std::vector<std::vector<T>> arr)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>(1);
		m_cells[0] = arr;
		dims = 1;
		rows = arr.size();
		cols = arr[0].size();
	}
	matrix<T>(std::vector<T> arr)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>(1);
		for (int i = 0; i < arr.size(); ++i)
		{
			std::vector<T> cell;
			cell.push_back(arr[i]);
			m_cells[0].push_back(cell);
		}
		dims = 1;
		rows = arr.size();
		cols = 1;
	}
	~matrix<T>()
	{
	}
	unsigned int rows;
	unsigned int cols;
	unsigned int dims;
	T at(unsigned int i, unsigned int j, unsigned int k)
	{
		return m_cells[k][i][j];
	}
	void set(unsigned int i, unsigned int j, unsigned int k, T value)
	{
		m_cells[k][i][j] = value;
	}
	matrix<T> at_channel(unsigned int k)
	{
		matrix result;
		result = m_cells[k];
		return result;
	}
	matrix<T> from(unsigned int left, unsigned int top, unsigned int width, unsigned int height)
	{
		matrix<T> sample(width, height, dims);

		for (int k = 0; k < dims; ++k)
		for (int i = top; i < top + height; ++i)
		for (int j = left; j < left + width; ++j)
			sample.set(i - top, j - left, k, (*this).at(i, j, k));
		return sample;
	}
	matrix<T> transpose()
	{
		matrix<T> result(rows, cols, dims);
		for (int k = 0; k < dims; ++k)
		for (int i = 0; i < cols; ++i)
		for (int j = 0; j < rows; ++j)
			result.set(i, j, k, at(j, i, k));
		return result;
	}
	std::string to_string()
	{
		std::stringstream ss;
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
				ss << at(i, j, 0) << " ";
			ss << "\n";
		}
		return ss.str();
	}
	std::vector<T> at_row(unsigned int i, unsigned int k)
	{
		return m_cells[k][i];
	}
	void set_row(unsigned int i, unsigned int k, std::vector<T> row_value)
	{
		m_cells[k][i] = row_value;
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
		rows = arr[0].size();
		cols = arr[0][0].size();
		return *this;
	}
	matrix<T> operator=(std::vector<std::vector<T>> arr)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>(1);
		m_cells[0] = arr;
		dims = 1;
		rows = arr.size();
		cols = arr[0].size();
		return *this;
	}
	matrix<T> operator=(std::vector<T> arr)
	{
		m_cells = std::vector<std::vector<std::vector<T>>>(1);
		m_cells[0].push_back(std::vector<std::vector<T>>(arr.size()));
		for (int i = 0; i < arr.size(); ++i)
		{
			std::vector<T> cell;
			cell.push_back(arr[i]);
			m_cells[0][i].push_back(cell);
		}
		dims = 1;
		rows = arr.size();
		cols = 1;
		return *this;
	}
	matrix<T> operator*(T scalar)
	{
		for (int k = 0; k < this->dims; ++k)
		for (int i = 0; i < this->rows; ++i)
		for (int j = 0; j < this->cols; ++j)
			this->set(i, j, k, this->at(i, j, k) * scalar);
		return *this;
	}
	bool operator==(matrix<T> other)
	{
		for (int k = 0; k < this->dims; ++k)
		for (int i = 0; i < this->rows; ++i)
		for (int j = 0; j < this->cols; ++j)
		if (this->at(i, j, k) != other.at(i, j, k))
			return false;
		return true;
	}
private:
	std::vector<std::vector<std::vector<T>>> m_cells;
};

