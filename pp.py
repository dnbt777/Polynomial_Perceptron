import numpy as np
import random
import time

# taylor
#f(x)≈f(a)+f′(a)(x−a)+2!f′′(a)​(x−a)2+3!f′′′(a)​(x−a)3+⋯

# mclaurin
# f(x)≈f(a)+f′(a)(x−a)+2!f′′(a)​(x−a)2+3!f′′′(a)​(x−a)3+⋯
# a=0



def MSE(y, yhat):
    return (y - yhat)**2

def MSE_derivative(y, yhat):
    return 2*(y - yhat) # (abs?)

def softmax(logits):
    # Subtract the max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

class PP():
    def __init__(self, io, constants, pp_type="exp", eta=1e-4, pp_softmax=True):
        self.a = np.zeros((io[0],)) # a in taylor series, set to 0 for now (mclaurin)
        self.type = pp_type # mclaurin taylor simple_exp exp
        # exp = z(n=0->m) exp(w*x^n), df_dw = x^n * exp(w * x^n)
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



    def __repr__(self):
        reprstring = "f(x) = "
        constant_strings = [str(constant.shape) for constant in self.constants[1:]]
        neuron_strings = [str(neuron.shape) for neuron in self.neurons]
        
        if self.type == "exp":
            terms = [f"exp({self.constants[0].shape})"]
            terms += [f"exp({constant}*{neuron}^{i+1})" for constant, neuron, i in zip(constant_strings, neuron_strings, range(len(neuron_strings)))]
        else:
            terms = [f"{self.constants[0].shape}"]
            terms += [f"{constant}*{neuron}^{i+1}" for constant, neuron, i in zip(constant_strings, neuron_strings, range(len(neuron_strings)))]
            
        terms = " + ".join(terms)
        return reprstring + terms


    def forward(self, x):
        if self.softmax:
            x = softmax(x)/len(x)
        terms = []
        term = self.constants[0]
        if self.type == "exp":
                term = np.exp(term)
        terms.append(self.constants[0])
        for i, constant_matrix in enumerate(self.constants[1:]):
            x_component = np.power(x, i) # unoptimized but who cares its mvp # if mclaurin add a or whatever
            term = constant_matrix @ x_component
            if self.type == "exp":
                term = np.exp(term) # approximates any function in one term, should converge w less terms
            assert term.shape == (self.output_shape,), "shape mismatch"
            terms.append(term)
        yhat = np.sum(np.array(terms), axis=0)
        self.last_terms = terms
        self.last_x = x
        return yhat


    def backward(self, yhat, y):
        dL_df = MSE_derivative(y, yhat)
        
        if self.type == "mclaurin":
            for i, c in enumerate(self.constants):
                if i == 0:
                    df_dc = 1
                    self.weight_updates.append(dL_df)
                else:
                    df_dc = np.power(self.last_x, i) # very clean how the index lines up w the exponent lol
                    dL_dc = np.outer(dL_df, df_dc)
                    self.weight_updates.append(dL_dc)
        

        if self.type == "exp":
            for i, c in enumerate(self.constants):
                if i == 0:
                    # f = exp(w), df_dw = exp(w)
                    df_dc = np.exp(c)
                    self.weight_updates.append(dL_df*df_dc) # elementwise
                else:
                    xpow = np.power(self.last_x, i)
                    #df_dc = np.exp(c @ xpow)*xpow # very clean how the index lines up w the exponent lol - also could store exp part in forward for speedup
                    #dL_dc = dL_df*df_dc

                    dL_dc = np.outer(dL_df*np.exp(c @ xpow), xpow) # diff order of operations
                    self.weight_updates.append(dL_dc)    


        if self.grad_clipping is not None:
            self.weight_updates = [np.clip(update, -self.grad_clipping, self.grad_clipping) for update in self.weight_updates]


    def update_and_zero_grad(self):
        for i, update in enumerate(self.weight_updates):
            self.constants[i] -= update * self.eta

        self.last_x = None
        self.weight_updates = []


    



# simple version - elementwise multiplication of x and x
# complex version - outer product
# f(x) = a + bx + cx^2 + dx^3 + ex^4
# f(x) shape 11,
# x shape 2,


# b*x = (11, )
# c*x*x = (2, ) * (2, ) * (2, 11)
# each constant is (input, output) shape. xs are multiplied element wise.

# f(x) = a + x*(b + x*(c + x*(d + x*(e))))


# f(x) <= (11,)
# a <= (11,)
# x*(b + ...) <= (11, )
# x <= (2, )
# (b + x*...) <= (2, 11)


def testpp(pptype):
    io = [1, 1]
    constants = 4 # a, b, c, d, etc -
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


if __name__ == "__main__":
    testpp("exp")
    testpp("mclaurin")


