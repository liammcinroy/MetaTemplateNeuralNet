#pragma once

#include <array>
#include <initializer_list>

#include <math.h>
#include <stdlib.h>

template<typename T> class Matrix
{
public:
	Matrix<T>() = default;

	virtual ~Matrix<T>() = default;

	virtual inline T& at(unsigned int i, unsigned int j) = 0;

	virtual inline T* row(unsigned int i) = 0;

	virtual inline T* col(unsigned int j) = 0;

	virtual void elem_multiply(Matrix<T>* &other)
	{
	}

	virtual void elem_divide(Matrix<T>* &other)
	{
	}

	virtual inline unsigned int rows() = 0;

	virtual inline unsigned int cols() = 0;
};

template<typename T, unsigned int r, unsigned int c> class Matrix2D : public Matrix<T>
{
private:
	std::array<T, r * c> data;
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
		std::initializer_list<std::initializer_list<T>>::iterator it = arr.begin();
		for (int i = 0; i < rows; ++i)
		{
			std::initializer_list<T>::iterator it2 = it->begin();
			for (int j = 0; j < cols; ++j)
			{
				data[(cols * i) + j] = *it2;
				++it2;
			}
			++it;
		}
	}

	Matrix2D<T, r, c>(std::initializer_list<T> arr)
	{
		std::initializer_list<T>::iterator it = arr.begin();
		for (int i = 0; i < r; ++i)
		{
			data[(c * i)] = *it;
			++it;
		}
	}

	~Matrix2D<T, r, c>() = default;

	inline T& at(unsigned int i, unsigned int j) 
	{
		return data[(c * i) + j];
	}

	inline T* row(unsigned int i) 
	{
		T* output = new T[c];
		for (unsigned int j = c * i; j < c * i + c; ++j)
			output[j - c * i] = data[j];
		return output;
	}

	inline T* col(unsigned int j) 
	{
		T* output = new T[r];
		for (unsigned int i = j; i < r * c; i += c)
			output[(i - j) / c] = data[i];
		return output;
	}

	void elem_multiply(Matrix2D<T, r, c>* &other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
					this->at(i, j) *= other->at(i, j);
	}

	void elem_divide(Matrix2D<T, r, c>* &other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
					this->at(i, j) /= other->at(i, j);
	}

	Matrix2D<T, r, c> operator+(Matrix2D<T, r, c> &other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) += other.at(i, j);
		return *this;
	}

	Matrix2D<T, r, c> operator-(Matrix2D<T, r, c> &other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) -= other.at(i, j);
		return *this;
	}

	Matrix2D<T, r, c> operator*(T &scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) *= scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator/(T &scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) /= scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator+(T &scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) += scalar;
		return *this;
	}

	Matrix2D<T, r, c> operator-(T &scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) -= scalar;
		return *this;
	}

	inline unsigned int rows() 
	{ 
		return r;
	}

	inline unsigned int cols() 
	{
		return c;
	}
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
			for (int i2 = 0; i2 < rows; ++i2)
				sum += lhs.at(i, i2) * rhs.at(i2, j);
			result.at(i, j) = sum;
		}
	}
	return result;
}

template<typename T, unsigned int length> class Vector : public Matrix2D<T, length, 1>
{
public:
	T operator*(Vector<T, length> &other)
	{
		T sum();
		for (int i = 0; i < length; ++i)
			sum += (this->at(i, 0) * other.at(i, 0));
	}
};