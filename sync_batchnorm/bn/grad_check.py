from __future__ import division
import torch
from bn import func_bn
from torch.autograd import gradcheck, Variable
import torch.nn as nn
import time

if __name__ == '__main__':
    input = torch.rand(10,30,10).double().cuda()
    input.requires_grad = True
    mean = torch.rand(30).double().cuda()
    mean.requires_grad = True
    inv_std = torch.rand(30).double().cuda()
    inv_std.requires_grad = True
    gamma = torch.rand(30).double().cuda()
    gamma.requires_grad = True
    beta = torch.rand(30).double().cuda()
    beta.requires_grad = True
    func = func_bn()

    input2 = input.clone()
    mean2 = mean.clone()
    inv_std2 = inv_std.clone()
    gamma2 = gamma.clone()
    beta2 = beta.clone()
    func2 = func_bn()
    mean2, inv_std2, gamma2, beta2 = [x.unsqueeze(0).unsqueeze(-1) for x in [mean2, inv_std2, gamma2, beta2] ]
    print('shape: {}'.format(mean2.shape))
    out = func2(input2, mean2, inv_std2, gamma2, beta2)
    print('out shape: {}'.format(out.shape))
    out_true = (input2 - mean2) * inv_std2 * gamma2 + beta2
    print('out_true shape: {}'.format(out_true.shape))
    print('diff sum: {}'.format((out - out_true).sum().item()))
    print('diff num: {}'.format(((out - out_true).abs() > 1e-6).sum().item()))

    #input = torch.rand(3, 32, 48, 200, 200).float().cuda()
    #input.requires_grad = True
    #input = input.view(3,32,-1)
    #mean = torch.rand(32).float().cuda()
    #mean.requires_grad = True
    #inv_std = torch.rand(32).float().cuda()
    #inv_std.requires_grad = True
    #gamma = torch.rand(32).float().cuda()
    #gamma.requires_grad = True
    #beta = torch.rand(32).float().cuda()
    #beta.requires_grad = True
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #begin = time.time()
    #for i in range(10):
    #    out = (input - mean.unsqueeze(0).unsqueeze(-1)) * inv_std.unsqueeze(0).unsqueeze(-1) * gamma.unsqueeze(0).unsqueeze(-1) + beta.unsqueeze(0).unsqueeze(-1)
    #    out.sum().backward()
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #end = time.time()
    #print('imperative time: {}'.format(end - begin))
    #print(mean.shape)

    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #begin = time.time()
    #for i in range(10):
    #    out = func_bn()(input, mean, inv_std, gamma, beta)
    #    out.sum().backward()
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #end = time.time()
    #print('custom time: {}'.format(end - begin))

    if gradcheck(func, [input, mean, inv_std, gamma, beta]):
        print('OK')
    else:
        print('FAIL')
