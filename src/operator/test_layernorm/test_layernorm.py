import mxnet as mx
import numpy as np
import time
from mxnet.gluon import nn

dry_run = 50
num_iter = 1000

def test_layernorm(shape):

    np_data = np.random.rand(shape[0], shape[1], shape[2])
    x = mx.nd.array(np_data)
    y = mx.nd.array(np_data)
    gamma = mx.nd.array([1.0])
    beta = mx.nd.array([0.0])
    mx.nd.waitall()
    for i in range(dry_run+num_iter):
        if i == dry_run:
            tic = time.time()
        y = mx.nd.LayerNorm(x, gamma, beta, axis=-1)
        y.wait_to_read()
    
    return time.time() - tic


if __name__ == "__main__":

    shapes = [[1, 128, 768], [8, 128, 768], [32, 128, 768], [1, 128, 1024], [8, 128, 1024], [32, 128, 1024]]
    #shapes = [[1, 128, 768]]
    for shape in shapes:
        cost = test_layernorm(shape)
        print("shape: %s, \ttime: %.6f ms" %( str(shape), (cost / num_iter * 1000) ))
