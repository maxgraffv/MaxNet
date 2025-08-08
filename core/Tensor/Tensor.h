#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor
{

    private:
        std::vector<float> _data;       
        std::vector<int> _shape;       
        std::vector<int> _strides;       
        int _size;
        void initStrides();

        

    public:
        Tensor(std::vector<int> shape);


        /*
            Access
        */
        size_t linearIndex(const std::vector<size_t>& indices) const;

        float& operator[](size_t index);
        float& at(std::vector<size_t> indices);
        const float& at(std::vector<size_t> indices) const;

        int size();
        const std::vector<int> shape() const;


        /*
            Mathematical
        */
        Tensor add(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        float dot(const Tensor& other) const;
};


#endif