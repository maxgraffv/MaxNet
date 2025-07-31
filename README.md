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
- Tensor
- Device
- Shape, Stride, Broadcasting
- AutogradEngine

### Operations
- add
- mul
- matmul
- dot
- conv2d
- relu
- sigmoid

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






