from __future__ import division
import torch
from sum_square import func_sum_square
from torch.autograd import gradcheck, Variable
import time

if __name__ == '__main__':
    fea = torch.rand(10, 30, 10).double().cuda()
    fea.requires_grad = True
    func = func_sum_square()

    func2 = func_sum_square()
    fea2 = torch.tensor([1,2,3,4,4,5,6,7,2,4,5,7]).view(6,1,2).double().cuda()
    out1, out2 = func2(fea2)
    sum, ssum = fea2.sum(0).sum(-1), (fea2 ** 2).sum(0).sum(-1)
    print('out1: {}, out2: {}'.format(out1.item(), out2.item()))
    print('sum: {}, ssum: {}'.format(sum.item(), ssum.item()))
    print('diff sum: {}'.format(out1 - sum))
    print('diff ssum: {}'.format(out2 - ssum))

    #fea = torch.rand(3, 32, 48, 200, 200).float().cuda()
    #fea = fea.view(3,32,-1)
    #fea.requires_grad = True
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #begin = time.time()
    #for i in range(50):
    #    sum, ssum = fea.sum(0).sum(-1), (fea ** 2).sum(0).sum(-1)
    #    sum_all = sum.sum() + ssum.sum()
    #    sum_all.backward()
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #end = time.time()
    #print('imperative time: {}'.format(end - begin))
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #begin = time.time()
    #for i in range(50):
    #    sum, ssum = func_sum_square()(fea)
    #    sum_all = sum.sum() + ssum.sum()
    #    sum_all.backward()
    #torch.cuda.synchronize()
    #torch.cuda.synchronize()
    #end = time.time()
    #print('custom time: {}'.format(end - begin))


    #fea = torch.rand(1, 30, 10, 10).double().cuda()
    #fea[0, 10, 2, 2] = 100
    #print('before: {}'.format(fea[0,:,2,2]))
    #out = func(fea)
    #out.sum().backward()
    #print('after: {}'.format(out[0,:,2,2]))
    #print('after grad: {}'.format(fea.grad[0,:,2,2]))
    if gradcheck(func, [fea]):
        print('OK')
    else:
        print('FAIL')
