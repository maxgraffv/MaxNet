#include "Tensor.h"

#include <iostream>


Tensor::Tensor(std::vector<int> shape) : _shape(shape), _size(0)
{
    int temp_size = 1;
    for(int i = 0; i < _shape.size(); i++)
    {
        temp_size *= _shape[i];
    }
    _size = temp_size;

    initStrides();

    this->_data = std::vector<float>(_size);
}


int Tensor::size()
{
    return _size;
}

size_t Tensor::linearIndex(const std::vector<size_t>& indices) const
{
    if(indices.size() != _shape.size())
        throw std::invalid_argument("Inavlid number of indices");

    size_t linearInd = 0;
    for(size_t i = 0; i < _shape.size(); i++)
    {
        if(indices[i] >= _shape[i])
            throw std::out_of_range("Index out of bounds");
        
        linearInd += indices[i] * _strides[i];
    }

    return linearInd;
}


const std::vector<int> Tensor::shape() const
{
    return _shape;
}

float& Tensor::operator[](size_t index)
{
    return _data[index];
}


const float& Tensor::at(std::vector<size_t> indices) const
{
    return _data[linearIndex(indices)] ;
}

float& Tensor::at(std::vector<size_t> indices)
{
    return _data[linearIndex(indices)] ;
}

void Tensor::initStrides()
{
    _strides.resize(_shape.size());
    size_t stride = 1;

    for(int i = _shape.size()-1; i >= 0; --i)
    {
        _strides[i] = stride;
        stride *= _shape[i];
    }
}


Tensor Tensor::add(const Tensor& other) const
{
    if(this->_shape != other._shape)
        throw std::invalid_argument("Wrong Tensor shape for Add");

    Tensor result(_shape);
    for (int i = 0; i < _size; ++i) {
        result._data[i] = _data[i] + other._data[i];
    }
    return result;
}


Tensor Tensor::matmul(const Tensor& other) const
{
    if (_shape.size() != 2 || other._shape.size() != 2)
        throw std::invalid_argument("Only 2D tensors supported in matmul");

    int m = _shape[0];
    int n = _shape[1];
    int n2 = other._shape[0];
    int p = other._shape[1];

    if (n != n2)
        throw std::invalid_argument("Incompatible shapes for matmul");

    Tensor result({m, p});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += this->at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    return result;
}


Tensor Tensor::mul(const Tensor& other) const {
    if (_shape != other._shape)
        throw std::invalid_argument("Shape mismatch in mul");

    Tensor result(_shape);
    for (int i = 0; i < _size; ++i) {
        result._data[i] = _data[i] * other._data[i];
    }
    return result;
}


float Tensor::dot(const Tensor& other) const {
    if (_shape.size() != 1 || other._shape.size() != 1)
        throw std::invalid_argument("dot: tensors must be 1D");
    if (_shape[0] != other._shape[0])
        throw std::invalid_argument("dot: shape mismatch");

    float result = 0.0f;
    for (int i = 0; i < _size; ++i) {
        result += _data[i] * other._data[i];
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_size = 1;
    for (int dim : new_shape) {
        if (dim != -1) new_size *= dim;
    }

    int inferred = -1;
    if (std::find(new_shape.begin(), new_shape.end(), -1) != new_shape.end()) {
        inferred = _size / new_size;
    }

    std::vector<int> final_shape = new_shape;
    for (int& dim : final_shape) {
        if (dim == -1) dim = inferred;
    }

    if (_size != new_size * (inferred > 0 ? 1 : 0)) {
        throw std::runtime_error("Invalid reshape: total size mismatch");
    }

    Tensor result(final_shape);
    result._data = _data; // shallow copy of data
    return result;
}