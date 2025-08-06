# MaxNet
Image Recognition by CNN <br/>
CUDA assisted <br/>

Based on PyTorch opartions <br/>

<br/>
<br/>
<br/>

## PyTorch structure
<br/>

### Core
- [X] Tensor
- Device
- [X] Shape, Stride, 
- Broadcasting
- AutogradEngine

### Operations
- [X] add
- [X] mul
- [X] matmul
- [X] dot

- [X] conv2d
- [X] relu
- [X] sigmoid

### Layers
- Linear
- Conv2D
- MaxPool
- ReLU
- Softmax
- Flatten
- Sequential

## Model <br/>
Model model = Sequential({LAYERS array}) - as if a stream - model = input.Conv2D().ReLU().MaxPool()....Linear()

## Training
- loss = model.forward(input)
- loss.backward() - autograd
- optimizer.step() - AdamW



## Learn more about
- Dropout, BN, skip



