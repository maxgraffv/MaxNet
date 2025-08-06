#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Tensor.h"

class NeuralNet
{
    private:


    public:
        NeuralNet();

        Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride = 1,
            int padding = 0);

        Tensor relu(const Tensor& input);
        Tensor sigmoid(const Tensor& input);
        Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias);
        Tensor softmax(const Tensor& input);
        Tensor forward(const Tensor& input);

};


#endif