import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    s = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std = s, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = 1.0
    if(nonlinearity == "relu"):
        bound = math.sqrt(2.0) * math.sqrt(3.0/fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    s = 1.0
    if(nonlinearity == "relu"):
        s = math.sqrt(2.0) / math.sqrt(fan_in)
    return randn(fan_in, fan_out, std=s, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION
