"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad = True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        second_dim = 1
        for s in X.shape[1:]:
            second_dim *= s
        return ops.reshape(X, (X.shape[0],second_dim))
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        shape_z = logits.shape
        one_hot = init.one_hot(shape_z[len(shape_z)-1],y)
        z_y = one_hot * logits
        z_y = ops.summation(z_y, axes=(1,))
        out = ops.logsumexp(logits, axes=(1,))
        return ops.summation(out - z_y)/out.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.running_mean = Tensor(np.zeros(dim))
        self.running_var = Tensor(np.ones(dim))
        self.weight = Parameter(Tensor(np.ones(dim),dtype=dtype,device=device))
        self.bias = Parameter(Tensor(np.zeros(dim), device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == False:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var.broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        mu = ops.summation(x, (0,))/x.shape[0]
        self.running_mean = ops.mul_scalar(self.running_mean, 1-self.momentum) + ops.mul_scalar(mu, self.momentum)
        mu = ops.broadcast_to(mu, x.shape)
        var = ops.summation(ops.power_scalar(x + mu*-1, 2), axes=(0,))/ x.shape[0]
        self.running_var = ops.mul_scalar(self.running_var, 1-self.momentum) + ops.mul_scalar(var, self.momentum)
        var = ops.broadcast_to(var, x.shape)

        w = ops.broadcast_to(self.weight, x.shape)
        bias = ops.broadcast_to(self.bias, x.shape)
        out = w * (x - mu)/ops.power_scalar(var + self.eps,0.5) + bias
        return out
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(Tensor(np.ones(dim),dtype=dtype,device=device))
        self.bias = Parameter(Tensor(np.zeros(dim), device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = (x.shape[0], 1)
        mu = ops.summation(x,axes=(1,))/x.shape[1]
        mu = ops.broadcast_to(ops.reshape(mu, shape) , x.shape)
        var = ops.summation(ops.power_scalar(x + mu*-1, 2), axes=(1,))/ x.shape[1]
        var = ops.broadcast_to(ops.reshape(var,shape), x.shape)

        w = ops.broadcast_to(self.w, x.shape)
        bias = ops.broadcast_to(self.bias, x.shape)

        out = w * (x - mu)/ops.power_scalar(var + self.eps,0.5) + bias
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(*x.shape, p=1-self.p)
        mask = mask / (1.0-self.p)
        return x*Tensor(mask)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
