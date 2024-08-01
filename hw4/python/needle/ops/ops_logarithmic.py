from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_kd = array_api.max(Z, axis = self.axes, keepdims=True)
        max_z = array_api.max(Z, axis = self.axes)
        max_z_kd = array_api.broadcast_to(max_z_kd, Z.shape)
        sum_z = array_api.sum(array_api.exp(Z - max_z_kd), axis = self.axes)
        return array_api.log(sum_z) + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z, = node.inputs
        arr_z = z.numpy().copy()
        max_z = array_api.max(arr_z, axis = self.axes, keepdims=True)
        max_z = array_api.broadcast_to(max_z, z.shape)

        exp_z = array_api.exp(arr_z - max_z)
        sum_z = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        sum_z = array_api.broadcast_to(sum_z, z.shape)

        z_shape = list(z.shape)
        if self.axes != None:
            for s in self.axes:
                z_shape[s] = 1
            out_grad = reshape(out_grad, tuple(z_shape))
        return Tensor(exp_z/sum_z) * out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

