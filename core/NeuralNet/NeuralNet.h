#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Tensor.h"
#include <cmath>
#include <random>
#include <iostream>

class NeuralNet
{
    private:
       Tensor conv_weight;
       Tensor conv_bias;
       Tensor fc_weight;
       Tensor fc_bias;

    public:
        NeuralNet();

        /*
            Layers
        */
        Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride = 1,
            int padding = 0);

        Tensor relu(const Tensor& input);
        Tensor sigmoid(const Tensor& input);
        Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias);
        Tensor softmax(const Tensor& input);

        Tensor forward(const Tensor& input);
        float crossEntropyLoss(const Tensor& predictions, const Tensor& targets);
        void train(const std::vector<Tensor>& inputs,
                    const std::vector<Tensor>& targets,
                    int epochs = 10, float lRate = 0.001f);


        void NeuralNet::backward(const Tensor& input, const Tensor& output, 
                                const Tensor& target, float lr);
};


#endif