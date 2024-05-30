# Polynomial_Perceptron Network

This repository contains an implementation of a Polynomial Perceptron (PP) network, which can approximate functions using different types of polynomial expansions, including Taylor and McLaurin series. The network supports various types of polynomial expansions and can be used for function approximation tasks.

## Network Types

### Taylor Series

The Taylor series expansion of a function \( f(x) \) around a point \( a \) is given by:

$f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \cdots$

In the context of the PP network, the Taylor series can be represented as:

$f(x) \approx a_0 + a_1 \cdot (x - a) + a_2 \cdot (x - a)^2 + a_3 \cdot (x - a)^3 + \cdots$

Where:
- \( a_0 \) is the constant term.
- \( a_1, a_2, a_3, \ldots \) are the coefficients of the polynomial terms.
- \( x \) is the input to the network.

The network approximates the function by learning the coefficients \( a_0, a_1, a_2, \ldots \) through training.

### McLaurin Series

The McLaurin series is a special case of the Taylor series where \( a = 0 \):

$f(x) \approx f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots$

In the context of the PP network, the McLaurin series can be represented as:

$f(x) \approx a_0 + a_1 \cdot x + a_2 \cdot x^2 + a_3 \cdot x^3 + \cdots$

Where:
- \( a_0 \) is the constant term.
- \( a_1, a_2, a_3, \ldots \) are the coefficients (matrices) of the polynomial terms.
- \( x \) is the input to the network.

The network approximates the function by learning the coefficients \( a_0, a_1, a_2, \ldots \) through training.

### Exponential Taylor Series

The Exponential Taylor Series is an extension of the Taylor series where each term is an exponential function. This approach may have merit as it should theoretically allow each term to approximate more complex functions, including imaginary numbers and trigonometric functions. No idea if this is a thing already

The Exponential Taylor Series expansion of a function \( f(x) \) around a point \( a \) is given by:

$f(x) \approx \exp(a_0) + \exp(a_1 \cdot (x - a)) + \exp(a_2 \cdot (x - a)^2) + \exp(a_3 \cdot (x - a)^3) + \cdots$

Where:
- \( \exp(a_0) \) is the exponential of the constant term.
- \( \exp(a_1 \cdot (x - a)), \exp(a_2 \cdot (x - a)^2), \exp(a_3 \cdot (x - a)^3), \ldots \) are the exponential terms.
- \( x \) is the input to the network.

The network approximates the function by learning the coefficients \( a_0, a_1, a_2, \ldots \) through training.
