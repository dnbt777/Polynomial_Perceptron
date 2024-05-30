# Polynomial_Perceptron Network

This repository contains an implementation of a Polynomial Predictor (PP) network, which can approximate functions using different types of polynomial expansions, including Taylor and McLaurin series. The network supports various types of polynomial expansions and can be used for function approximation tasks.

## Network Types

### Taylor Series

The Taylor series expansion of a function \( f(x) \) around a point \( a \) is given by:

\[ f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \cdots \]

### McLaurin Series

The McLaurin series is a special case of the Taylor series where \( a = 0 \):

\[ f(x) \approx f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots \]

## Code Overview

The `PP` class implements the Polynomial Predictor network. Below is a brief overview of the key components:

### Initialization

The `__init__` method initializes the network with the given input-output dimensions, number of constants, type of polynomial expansion, learning rate, and whether to apply softmax to the inputs.

```python
class PP():
    def __init__(self, io, constants, pp_type="exp", eta=1e-4, pp_softmax=True):
        self.a = np.zeros((io[0],)) # a in taylor series, set to 0 for now (mclaurin)
        self.type = pp_type # mclaurin taylor simple_exp exp
        self.input_shape = io[0] # x
        self.output_shape = io[1] # f(x)
        self.neuron_shapes = [(io[0],) for n in range(1, constants - 1)]
        self.neurons = [np.zeros(neuron_shape) for neuron_shape in self.neuron_shapes]
        self.constants = [np.random.rand(self.output_shape)] + [np.random.rand(self.input_shape*self.output_shape).reshape((self.output_shape, self.input_shape)) for _ in range(len(self.neurons))]
        self.last_x = None
        self.weight_updates = []
        self.eta = eta
        self.softmax = pp_softmax
        self.grad_clipping = None
        self.last_terms = []
```

### Forward Pass

The `forward` method computes the output of the network for a given input `x`.

```python
def forward(self, x):
    if self.softmax:
        x = softmax(x)/len(x)
    terms = []
    term = self.constants[0]
    if self.type == "exp":
        term = np.exp(term)
    terms.append(self.constants[0])
    for i, constant_matrix in enumerate(self.constants[1:]):
        x_component = np.power(x, i)
        term = constant_matrix @ x_component
        if self.type == "exp":
            term = np.exp(term)
        assert term.shape == (self.output_shape,), "shape mismatch"
        terms.append(term)
    yhat = np.sum(np.array(terms), axis=0)
    self.last_terms = terms
    self.last_x = x
    return yhat
```

### Backward Pass

The `backward` method computes the gradients of the loss with respect to the network's parameters.

```python
def backward(self, yhat, y):
    dL_df = MSE_derivative(y, yhat)
    if self.type == "mclaurin":
        for i, c in enumerate(self.constants):
            if i == 0:
                df_dc = 1
                self.weight_updates.append(dL_df)
            else:
                df_dc = np.power(self.last_x, i)
                dL_dc = np.outer(dL_df, df_dc)
                self.weight_updates.append(dL_dc)
    if self.type == "exp":
        for i, c in enumerate(self.constants):
            if i == 0:
                df_dc = np.exp(c)
                self.weight_updates.append(dL_df*df_dc)
            else:
                xpow = np.power(self.last_x, i)
                dL_dc = np.outer(dL_df*np.exp(c @ xpow), xpow)
                self.weight_updates.append(dL_dc)
    if self.grad_clipping is not None:
        self.weight_updates = [np.clip(update, -self.grad_clipping, self.grad_clipping) for update in self.weight_updates]
```

### Update and Zero Gradients

The `update_and_zero_grad` method updates the network's parameters using the computed gradients and resets the gradients.

```python
def update_and_zero_grad(self):
    for i, update in enumerate(self.weight_updates):
        self.constants[i] -= update * self.eta
    self.last_x = None
    self.weight_updates = []
```

## Example Usage

The `testpp` function demonstrates how to use the `PP` class to approximate a function (e.g., cosine function).

```python
def testpp(pptype):
    io = [1, 1]
    constants = 4
    pp_type = pptype
    pp_softmax = True
    pp = PP(io, constants, pp_type=pp_type, pp_softmax=False)
    print(pp)
    datapoints = 1
    xs = [np.random.rand(io[0]) for _ in range(datapoints)]
    ys = [np.cos(x) for x in xs]
    data = list(zip(xs, ys))
    epochs = 100000
    avgrunlosses = []
    printevery=epochs/10
    start = time.time()
    for i in range(epochs):
        x, y = random.choice(data)
        yhat = pp.forward(x)
        loss = MSE(y, yhat)
        avgrunlosses.append(loss)
        pp.backward(y, yhat)
        pp.update_and_zero_grad()
        if i%printevery==0 and i!=0:
            print(f"loss: at {i}", np.average(avgrunlosses), f"time={time.time()-start:.2f}")
            avgrunlosses = []
```

## Running the Code

To run the code, simply execute the script:

```bash
python script_name.py
```

This will test the `PP` network with both "exp" and "mclaurin" types.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
