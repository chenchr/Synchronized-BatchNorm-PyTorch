import torch
import torch.nn as nn
import Queue

data = torch.rand(8,4,10,10).cuda()
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.name0 = 'xxx'
        self.name1 = 'xxx'
        self.qq = Queue.Queue(5)
        print(self.qq.maxsize)

    def forward(self, x):
        if x.get_device() == 0:
            print('device 0')
            self.name0 = 'bbb'
        elif x.get_device() == 1:
            print('device 1')
            self.name1 = 'jjj'
        self.qq.put(x.get_device())
        return x

net1 = net().cuda()
print('dict: \n {}'.format(net1.__dict__))
net1 = nn.DataParallel(net1)
out = net1(data)
print(net1.module.name0)
print(net1.module.name1)
while not net1.module.qq.empty():
    print(net1.module.qq.get())

temp = {'qq': Queue.Queue(), 'name0': 'nn', 'name1': 'kk'}
temp['qq'].put('xx')
temp['name0'] = 'temp'
temp2 = temp.copy()
temp2['qq'].put('jj')
temp2['name1'][-1] = 'x'
while not temp['qq'].empty():
    print(temp['qq'].get())
print(temp['name0'])
print(temp['name1'])
