#pragma once

#include <vector>
#include <initializer_list>
#include <memory>

#include <math.h>
#include <stdlib.h>

//basic abstract class - use for pointers to unknown sizes of Matrix2D (ie IMatrix<float>* m; ... m->at(i, j) ... )
template<typename T> class IMatrix
{
public:
    IMatrix() = default;

    ~IMatrix() = default;

    virtual T& at(const size_t& i, const size_t& j) = 0;

    virtual const T& at(const size_t& i, const size_t& j) const = 0;
};

//template class - used for references and ensures passing correct size matrices before runtime
template<typename T, size_t r, size_t c> class Matrix2D : public IMatrix<T>
{
public:

    //default constructor
    Matrix2D()
    {
        data = std::vector<T>(r * c);;
        for (size_t i = 0; i < r * c; ++i)
            data[i] = T();
    }

    //construct with all elements equal to same value (usually 0 or 1)
    Matrix2D(T val)
    {
        data = std::vector<T>(r * c);;
        for (size_t i = 0; i < r * c; ++i)
            data[i] = val;
    }

    //construct with all elements drawn randomly from uniform distribution (defined by params)
    Matrix2D(const T& min, const T& max)
    {
        data = std::vector<T>(r * c);;
        T diff = max - min;
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (diff * rand()) / RAND_MAX + min;
    }

    //deep copy
    Matrix2D(const Matrix2D<T, r, c>& ref)
    {
        data = std::vector<T>(r * c);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = ref.data[i];
    }

    //construct from particular example (doesn't work well with brace-initialization, hence commented out)
    /*Matrix2D(std::initializer_list<std::initializer_list<T>> arr)
    {
        data = std::vector<T>(r * c);;
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

    //for vectors
    Matrix2D(std::initializer_list<T> arr)
    {
        data = std::vector<T>(r * c);
        typename std::initializer_list<T>::iterator it = arr.begin();
        for (size_t i = 0; i < r; ++i)
        {
            data[(c * i)] = *it;
            ++it;
        }
    }*/

    //vector cleans itself up
    ~Matrix2D() = default;

    //get element
    T& at(const size_t& i, const size_t& j) override
    {
        return data[(c * i) + j];
    }

    //get element
    const T& at(const size_t& i, const size_t& j) const override
    {
        return data[(c * i) + j];
    }

    //deep copy - depreciated?
    Matrix2D clone()
    {
        Matrix2D<T, r, c> out = Matrix2D<T, r, c>();

        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j)
                out.at(i, j) = this->at(i, j);
        return out;
    }

    //will store result in current instance
    void elem_multiply(Matrix2D<T, r, c>& other)
    {
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j)
                this->at(i, j) *= other.at(i, j);
    }

    //will store result in current instance
    void elem_divide(Matrix2D<T, r, c>& other)
    {
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j)
                this->at(i, j) /= other.at(i, j);
    }

    //returns current rows (constexpr so no memory access!)
    static constexpr size_t rows()
    {
        return r;
    }

    //returns current cols (constexpr so no memory access!)
    static constexpr size_t cols()
    {
        return c;
    }

    //data, stored in vector so data is in heap, not stack
    std::vector<T> data;
};

//Basically just a vector of Matrix2D<>s
template<size_t f, size_t r, size_t c, typename T = float> class FeatureMap
{
public:

    //default constructor
    FeatureMap()
    {
        for (size_t k = 0; k < f; ++k)
            maps.push_back(Matrix2D<T, r, c>());
    }

    //set all to same value
    FeatureMap(T val)
    {
        for (size_t k = 0; k < f; ++k)
            maps.push_back(Matrix2D<T, r, c>(val));
    }

    //draw from uniform distribution (defined by params)
    FeatureMap(T max, T min)
    {
        for (size_t k = 0; k < f; ++k)
            maps.push_back(Matrix2D<T, r, c>(max, min));
    }

    //deep copy
    FeatureMap(const FeatureMap<f, r, c, T>& ref)
    {
        for (size_t k = 0; k < f; ++k)
            maps.push_back(ref[k]);
    }

    /*
    //from another, doesn't work well with brace initializers
    FeatureMap(std::initializer_list<Matrix2D<T, r, c>> arr)
    {
        typename std::initializer_list<Matrix2D<T, r, c>>::iterator it = arr.begin();
        for (size_t k = 0; k < f && it != arr.end(); ++k)
        {
            maps.push_back(Matrix2D<T, r, c>(*it));
            ++it;
        }
    }*/

    //vector takes care of itself
    ~FeatureMap() = default;

    //access the data
    Matrix2D<T, r, c>& operator[](const size_t& feat)
    {
        return maps[feat];
    }

    //access the data
    const Matrix2D<T, r, c>& operator[](const size_t& feat) const
    {
        return maps[feat];
    }

    //access the data
    Matrix2D<T, r, c>& at(const size_t& feat)
    {
        return maps[feat];
    }

    //returns current number of maps (constexpr so no memory access!)
    static constexpr size_t size()
    {
        return f;
    }

    //returns current rows (constexpr so no memory access!)
    static constexpr size_t rows()
    {
        return r;
    }

    //returns current cols (constexpr so no memory access!)
    static constexpr size_t cols()
    {
        return c;
    }

    //Can be useful in templates
    using type = Matrix2D<T, r, c>;

private:

    //vector so it's on heap
    std::vector<Matrix2D<T, r, c>> maps;
};

//basic matrix multiplication
template<typename T, size_t rows1, size_t cols1, size_t rows2, size_t cols2> Matrix2D<T, rows1, cols2> operator*(const Matrix2D<T, rows1, cols1>& lhs, const Matrix2D <T, rows2, cols2>& rhs)
{
    Matrix2D<T, rows1, cols2> result{};
    for (size_t i = 0; i < rows1; ++i)
    {
        for (size_t j = 0; j < cols2; ++j)
        {
            T sum{};
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
