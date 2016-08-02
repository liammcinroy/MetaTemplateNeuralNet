#pragma once

#include <array>
#include <initializer_list>

#include <math.h>
#include <stdlib.h>

template<typename T> class IMatrix
{
public:
	IMatrix<T>() = default;

	virtual ~IMatrix<T>() = default;

	virtual T& at(int i, int j) = 0;

	virtual IMatrix<T>* clone() = 0;

	virtual void elem_multiply(IMatrix<T>* &other)
	{
	}

	virtual void elem_divide(IMatrix<T>* &other)
	{
	}

	virtual int rows() const = 0;

	virtual int cols() const = 0;

};

template<typename T, int r, int c> class Matrix2D : public IMatrix<T>
{
public:
	Matrix2D<T, r, c>()
	{
		for (int i = 0; i < r * c; ++i)
			data[i] = T();
	}

	Matrix2D<T, r, c>(int min, int max)
	{
		int diff = max - min;
		for (int i = 0; i < data.size(); ++i)
			data[i] = (rand() % diff) + min;
	}

	Matrix2D<T, r, c>(std::initializer_list<std::initializer_list<T>> arr)
	{
		typename std::initializer_list<std::initializer_list<T>>::iterator it = arr.begin();
		for (int i = 0; i < r; ++i)
		{
			typename std::initializer_list<T>::iterator it2 = it->begin();
			for (int j = 0; j < c; ++j)
			{
				data[(c * i) + j] = *it2;
				++it2;
			}
			++it;
		}
	}

	Matrix2D<T, r, c>(std::initializer_list<T> arr)
	{
		typename std::initializer_list<T>::iterator it = arr.begin();
		for (int i = 0; i < r; ++i)
		{
			data[(c * i)] = *it;
			++it;
		}
	}

	~Matrix2D<T, r, c>() = default;

	T& at(int i, int j)
	{
		return data[(c * i) + j];
	}

	IMatrix<T>* clone()
	{
		Matrix2D<T, r, c>* out = new Matrix2D<T, r, c>();

		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				out->at(i, j) = this->at(i, j);

		return out;
	}

	void elem_multiply(Matrix2D<T, r, c>* &other)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) *= other->at(i, j);
	}

	void elem_divide(Matrix2D<T, r, c>* &other)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) /= other->at(i, j);
	}

	Matrix2D<T, r, c> operator+(Matrix2D<T, r, c> &other)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) += other.at(i, j);
		return *this;
	}

	Matrix2D<T, r, c> operator-(Matrix2D<T, r, c> &other)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) -= other.at(i, j);
		return *this;
	}

	Matrix2D<T, r, c> operator*(T &scalar)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) *= scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator/(T &scalar)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) /= scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator+(T &scalar)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) += scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator-(T &scalar)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				this->at(i, j) -= scalar;
		return *this;
	}

	int rows() const
	{
		return r;
	}

	int cols() const
	{
		return c;
	}

	std::array<T, r * c> data;
};

template<typename T, int rows1, int cols1, int rows2, int cols2> Matrix2D<T, rows1, cols2>
	operator*(Matrix2D<T, rows1, cols1>& lhs, Matrix2D <T, rows2, cols2>& rhs)
	{
		Matrix2D<T, rows1, cols2> result();
		for (int i = 0; i < rows1; ++i)
		{
			for (int j = 0; j < cols2; ++j)
			{
				T sum();
				for (int i2 = 0; i2 < rows2; ++i2)
					sum += lhs.at(i, i2) * rhs.at(i2, j);
				result.at(i, j) = sum;
			}
		}
		return result;
	}

//Adds two matricies, stores in the first, deletes the second
template<typename T, int rows, int cols> void add(IMatrix<T>* &first, IMatrix<T>* second)
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				first->at(i, j) += second->at(i, j);
		delete second;
	}

template<typename T, int rows, int cols> void add(IMatrix<T>* &first, IMatrix<T>* second, T mult)
{
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			first->at(i, j) += second->at(i, j) * mult;
	delete second;
}