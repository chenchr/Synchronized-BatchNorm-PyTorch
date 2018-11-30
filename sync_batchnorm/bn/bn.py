from __future__ import division
from torch.autograd import Function
import torch

import bn_cuda

class func_bn(Function):
    def __init__(self):
        super(func_bn, self).__init__()

    def forward(self, input, mean, inv_std, gamma, beta):
        '''
        input: n, c, num  3 dim tensor
        '''
        if not input.is_cuda:
            raise NotImplementedError
        assert(gamma.nelement() > 0 and beta.nelement() > 0)
        self.save_for_backward(input, mean, inv_std, gamma, beta)
        output, = bn_cuda.forward(input, mean, inv_std, gamma, beta)
        #mean, inv_std, gamma, beta = [x.view(1,-1,1) for x in [mean, inv_std, gamma, beta] ]
        #output = (input - mean) * inv_std * gamma + beta

        return output

    def backward(self, grad_out):
        if not grad_out.is_cuda:
            raise NotImplementedError

        input, mean, inv_std, gamma, beta = self.saved_tensors
        grad_out = grad_out.contiguous()

        grad_in, grad_mean, grad_inv_std, grad_gamma, grad_beta = \
            bn_cuda.backward(grad_out, input, mean, inv_std, gamma, beta, True)
        #mean, inv_std, gamma, beta = [x.view(1,-1,1) for x in [mean, inv_std, gamma, beta] ]
        #sum_out = grad_out.sum(0,keepdim=True).sum(-1,keepdim=True)
        #sum_inout = (input * grad_out).sum(0,keepdim=True).sum(-1,keepdim=True)
        #scale = gamma * inv_std
        #grad_mean = - scale * sum_out
        #temp = sum_inout - mean * sum_out
        #grad_inv_std = gamma * temp
        #grad_gamma = inv_std * temp
        #grad_beta = sum_out
        #grad_in = grad_out * scale
        #grad_mean, grad_inv_std, grad_gamma, grad_beta = [x.view(-1) for x in [grad_mean, grad_inv_std, grad_gamma, grad_beta] ]

        return grad_in, grad_mean, grad_inv_std, grad_gamma, grad_beta
