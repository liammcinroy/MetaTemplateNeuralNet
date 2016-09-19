#pragma once

#include <array>
#include <initializer_list>
#include <memory>

#include <math.h>
#include <stdlib.h>

template<typename T> class IMatrix
{
public:
	IMatrix() = default;

	~IMatrix() = default;

	virtual T& at(const size_t& i, const size_t& j) = 0;

	virtual const T& at(const size_t& i, const size_t& j) const = 0;
};

template<typename T, size_t r, size_t c> class Matrix2D : public IMatrix<T>
{
public:
	Matrix2D()
	{
		for (size_t i = 0; i < r * c; ++i)
			data[i] = T();
	}

	Matrix2D(T val)
	{
		for (size_t i = 0; i < r * c; ++i)
			data[i] = val;
	}

	Matrix2D(const T& min, const T& max)
	{
		T diff = max - min;
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = (diff * rand()) / RAND_MAX + min;
	}

	Matrix2D(std::initializer_list<std::initializer_list<T>>& arr)
	{
		typename std::initializer_list<std::initializer_list<T>>::iterator it = arr.begin();
		for (size_t i = 0; i < r; ++i)
		{
			typename std::initializer_list<T>::iterator it2 = it->begin();
			for (size_t j = 0; j < c; ++j)
			{
				data[(c * i) + j] = *it2;
				++it2;
			}
			++it;
		}
	}

	Matrix2D(std::initializer_list<T>& arr)
	{
		typename std::initializer_list<T>::iterator it = arr.begin();
		for (size_t i = 0; i < r; ++i)
		{
			data[(c * i)] = *it;
			++it;
		}
	}

	~Matrix2D() = default;

	T& at(const size_t& i, const size_t& j) override
	{
		return data[(c * i) + j];
	}

	const T& at(const size_t& i, const size_t& j) const override
	{
		return data[(c * i) + j];
	}

	Matrix2D clone()
	{
		Matrix2D<T, r, c> out = Matrix2D<T, r, c>();

		for (size_t i = 0; i < r; ++i)
			for (size_t j = 0; j < c; ++j)
				out.at(i, j) = this->at(i, j);
		return out;
	}

	void elem_multiply(Matrix2D<T, r, c>& other)
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				this->at(i, j) *= other.at(i, j);
	}

	void elem_divide(Matrix2D<T, r, c>& other)
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				this->at(i, j) /= other.at(i, j);
	}

	static constexpr size_t rows()
	{
		return r;
	}

	static constexpr size_t cols()
	{
		return c;
	}

	std::array<T, r * c> data;
};

template<size_t f, size_t r, size_t c, typename T = float> class FeatureMap
{
public:
	FeatureMap() = default;

	FeatureMap(T val)
	{
		for (int k = 0; k < f; ++k)
			maps[k] = Matrix2D<T, r, c>(val);
	}

	FeatureMap(T max, T min)
	{
		for (int k = 0; k < f; ++k)
			maps[k] = Matrix2D<T, r, c>(max, min);
	}

	~FeatureMap() = default;

	Matrix2D<T, r, c>& operator[](const size_t& feat)
	{
		return maps[feat];
	}
	Matrix2D<T, r, c>& at(const size_t& feat)
	{
		return maps[feat];
	}
	static constexpr size_t size()
	{
		return f;
	}
	static constexpr size_t rows()
	{
		return r;
	}
	static constexpr size_t cols()
	{
		return c;
	}

	using type = Matrix2D<T, r, c>;

private:
	std::array<Matrix2D<T, r, c>, f> maps;
};

template<typename T, size_t rows1, size_t cols1, size_t rows2, size_t cols2> Matrix2D<T, rows1, cols2> operator*(const Matrix2D<T, rows1, cols1>& lhs, const Matrix2D <T, rows2, cols2>& rhs)
{
	Matrix2D<T, rows1, cols2> result{};
	for (size_t i = 0; i < rows1; ++i)
	{
		for (size_t j = 0; j < cols2; ++j)
		{
			T sum();
			for (size_t i2 = 0; i2 < rows2; ++i2)
				sum += lhs.at(i, i2) * rhs.at(i2, j);
			result.at(i, j) = sum;
		}
	}
	return result;
}

//Adds two matricies, stores in the first
template<typename T, size_t rows, size_t cols> void add(Matrix2D<T, rows, cols>& first, const Matrix2D<T, rows, cols>& second)
{
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			first.at(i, j) += second.at(i, j);
}

//Adds two matricies, but first second is multiplied by mult, stores in the first
template<typename T, size_t rows, size_t cols> void add(Matrix2D<T, rows, cols>& first, const Matrix2D<T, rows, cols>& second, const T& mult)
{
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			first.at(i, j) += second.at(i, j) * mult;
}