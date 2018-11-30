from __future__ import division
from torch.autograd import Function
import torch

import sum_square_cuda

class func_sum_square(Function):
    def __init__(self):
        super(func_sum_square, self).__init__()

    def forward(self, fea):
        '''
        fea: n, c, num  3 dim tensor
        '''
        if not fea.is_cuda:
            raise NotImplementedError

        fea = fea.contiguous()

        self.save_for_backward(fea)
        sum, ssum = sum_square_cuda.forward(fea)
        #sum, ssum = fea.sum(0).sum(-1), (fea ** 2).sum(0).sum(-1)

        return sum, ssum

    def backward(self, grad_sum, grad_ssum):
        if not (grad_sum.is_cuda or grad_ssum.is_cuda):
            raise NotImplementedError

        fea, = self.saved_tensors
        grad_sum = grad_sum.contiguous()
        grad_ssum = grad_ssum.contiguous()

        grad_in, = sum_square_cuda.backward(grad_sum, grad_ssum, fea)
        #grad_sum, grad_ssum = [x.unsqueeze(0).unsqueeze(-1) for x in [grad_sum, grad_ssum] ]
        #grad_in = grad_sum + 2 * fea * grad_ssum

        return grad_in
