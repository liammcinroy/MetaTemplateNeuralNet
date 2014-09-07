#pragma once

#include <math.h>
#include <stdlib.h>

template<class T> class IMatrix
{
public:
	IMatrix()
	{
	}
	~IMatrix<T>()
	{
		delete[] data;
	}
	virtual T& at(int i, int j)
	{
		return data[(cols * i) + j];
	}
	virtual T* row(unsigned int i)
	{
		T* output = new T[cols];
		for (unsigned int j = cols * i; j < cols * i + cols; ++j)
			output[j - cols * i] = data[j];
		return output;
	}
	virtual T* col(unsigned int j)
	{
		T* output = new T[rows];
		for (unsigned int i = j; i < rows * cols; i += cols)
			output[(i - j) / cols] = data[i];
		return output;
	}
	unsigned int rows;
	unsigned int cols;
	unsigned int dims;
	T* data;
};

template<class T> class Matrix2D : public IMatrix<T>
{
public:
	Matrix2D<T>()
	{
	}
	Matrix2D<T>(unsigned int r, unsigned int c)
	{
		data = new T[rows * cols * sizeof(T)];
		rows = r;
		cols = c;
		dims = 1;
	}
	~Matrix2D<T>()
	{
		delete[] data;
	}
	virtual T& at(unsigned int i, unsigned int j)
	{
		return data[(cols * i) + j];
	}
	virtual T* row(unsigned int i)
	{
		T* output = new T[cols];
		for (unsigned int j = cols * i; j < cols * i + cols; ++j)
			output[j - cols * i] = data[j];
		return output;
	}
	virtual T* col(unsigned int j)
	{
		T* output = new T[rows];
		for (unsigned int i = j; i < rows * cols; i += cols)
			output[(i - j) / cols] = data[i];
		return output;
	}
	Matrix2D<T> from(unsigned int i1, unsigned int j1, unsigned int i2, unsigned int j2)
	{
		Matrix2D<T> output(i2 - i1, j2 - j1);
		int start = (cols * i1) + j1;
		int end = (cols * j2) + j2;
		for (unsigned int i = start; i < end; ++i)
			output.at((i - (i % cols)) / cols, i % cols) = data[i];
		return output;
	}
	void elem_multiply(Matrix2D<T> other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
					this->at(i, j) *= other.at(i, j);
	}
	void elem_divide(Matrix2D<T> other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
					this->at(i, j) /= other.at(i, j);
	}
	Matrix2D<T> operator+(Matrix2D<T> other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) += other.at(i, j);
		return *this;
	}
	Matrix2D<T> operator-(Matrix2D<T> other)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) -= other.at(i, j);
		return *this;
	}
	Matrix2D<T> operator*(T scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) *= scalar;
		return *this;
	}
	Matrix2D<T> operator/(T scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) /= scalar;
		return *this;
	}
	Matrix2D<T> operator+(T scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) += scalar;
		return *this;
	}
	Matrix2D<T> operator-(T scalar)
	{
		for (unsigned int i = 0; i < rows; ++i)
			for (unsigned int j = 0; j < cols; ++j)
				this->at(i, j) -= scalar;
		return *this;
	}
	Matrix2D<T> operator*(Matrix2D<T> other)
	{
		Matrix2D<T> result(rows, other.cols);
		for (unsigned int i = 0; i < rows; ++i)
		{
			for (unsigned int j = 0; j < other.cols; ++j)
			{
				T sum;
				for (int i2 = 0; i2 < rows; ++i2)
					sum += this->at(i, i2) * other.at(i2, j);
				result.at(i, j) = sum;
			}
		}
		return result;
	}
};

template<class T> class Matrix3D : public IMatrix<T>
{
public:
	Matrix3D<T>()
	{
	}
	Matrix3D<T>(int r, int c, int d)
	{
		rows = r;
		cols = c;
		dims = d;
		data = new T[r * c * d * sizeof(T)];
	}
	~Matrix3D<T>()
	{
		delete[] data;
	}
	virtual T& at(int i, int j, int k)
	{
		return data[rows * cols * k + cols * i + j];
	}
	virtual T* row(int i, int k)
	{
		T* output = new T[cols];
		for (int j = 0; j < rows; ++j)
			output[j] = this->at(i, j, k);
		return output;
	}
	virtual T* col(int j, int k)
	{
		T* output = new T[rows];
		for (int i = 0; i < rows; ++i)
			output[i] = this->at(i, j, k);
		return output;
	}
	Matrix3D<T> from(int i1, int j1, int i2, int j2)
	{
		Matrix3D<T> output(i2 - i1, j2 - j1, dims);
		for (int k = 0; k < dims; ++k)
			for (int i = i1; i < i2; ++i)
				for (int j = j1; j < j2; ++j)
					output.at(i - i1, j - j1, k) = this->at(i, j, k);
		return output;
	}
	void elem_multiply(Matrix3D<T> other)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) *= other.at(i, j, k);
	}
	void elem_divide(Matrix3D<T> other)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) /= other.at(i, j, k);
	}
	Matrix3D<T> operator+(Matrix3D<T> other)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) += other.at(i, j, k);
		return *this;
	}
	Matrix3D<T> operator-(Matrix3D<T> other)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) -= other.at(i, j, k);
		return *this;
	}
	Matrix3D<T> operator*(T scalar)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) *= scalar;
		return *this;
	}
	Matrix3D<T> operator/(T scalar)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) /= scalar;
		return *this;
	}
	Matrix3D<T> operator+(T scalar)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) += scalar;
		return *this;
	}
	Matrix3D<T> operator-(T scalar)
	{
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					this->at(i, j, k) -= scalar;
		return *this;
	}
};