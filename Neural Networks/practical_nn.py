import numpy as np

# First exercise.

a = b = c = d = e = f = g = h = i = 0.5

X = [(-1, -1), (1, -1)]
Y = [(1, 1), (2, -2)]

eta = 0.1
weight_decay = 0.05

# Accumulate the sum of the gradients wrt the weights
# (needed for batch mode).
sum_grad_c = 0

# Backpropagation in batch mode for one epoch.
for x, y in zip(X, Y):
    x1 = x[0]
    x2 = x[1]
    y1 = y[0]
    y2 = y[1]

    # Forward pass.
    s1 = a + b*x1 + c*x2
    z1 = np.math.tanh(s1)
    s2 = f + d*x2
    z2 = np.math.tanh(s2)
    s3 = i + h*z2 + g*z1
    z3 = s3
    o1 = z1
    o2 = z3

    # Backward pass.
    grad_o1 = 2*(o1 - y1)  # derivative of L wrt o1
    grad_o2 = 2*(o2 - y2)  # derivative of L wrt o2
    grad_z3 = grad_o2
    grad_s3 = grad_z3
    grad_z1 = grad_o1 + grad_s3*g

    # z1 = tanh(s1)  =>  dL/ds1 = dL/dz1 * tanh'(s1) = dL/dz1 * (1 - z1**2)
    grad_s1 = grad_z1 * (1-z1**2)

    # s1 = a + b*x1 + c*x2  =>  dL/dc = dL/s1 * ds1/dc
    grad_c = grad_s1 * x2 

    # Accumulate the gradient.
    # In online mode we would update c right away.
    sum_grad_c += grad_c
    
    # For the complete backpropagation algorithm, we would need to collect
    # the gradients wrt all the other weights, not only c.

# Update c in batch mode.
# Weight decay is equivalent to a ridge penalty:
# The objective is now F = L + lbd*(a**2 + ... + i**2) => dF/dc = dL/dc + 2*lbd*c.
c = (1 - 2*eta*weight_decay)*c - eta*sum_grad_c

print(c)


# Second exercise.

a = b = c = d = f = g = h = 0.5
f_init = 1 
delta_f = f - f_init

x1 = x2 = y = 1

eta = 0.5
momentum = 0.7

logistic = lambda s: 1/(1+np.exp(-s))

# Forward pass.
s1 = a + b*x1
z1 = logistic(s1)
s2 = d + c*x2
z2 = logistic(s2)
s3 = h + f*z1 + g*z2
o = logistic(s3)

# Backward pass.
absolute_loss = True

if absolute_loss:
    # (Sub)gradient for absolute loss.
    grad_o = 1 if o >= y else -1
else:
    # Gradient for squared loss.
    grad_o = 2*(o - y)  # derivative of L wrt o1

# z3 = tanh(s3)  =>  dL/ds3 = dL/do * sigmoid'(s3) = dL/do * o*(1 - o)
grad_s3 = grad_o * o * (1-o)

# s3 = h + f*z1 + g*z2  =>  dL/df = dL/s3 * ds3/df
grad_f = grad_s3 * z1 

# Update f with a momentum term.
f = f - eta*grad_f + momentum*delta_f

print(f)
