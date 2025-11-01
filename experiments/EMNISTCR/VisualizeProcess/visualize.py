import torch,os
import numpy as np

from torch.nn import Module
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

LOSSF = torch.nn.CrossEntropyLoss()
XDIR = './x_dir.states.pth'
YDIR = './y_dir.states.pth'

def plot_contour(func, x_range, y_range):
    """
    创建一个等高线图（contour plot）来可视化给定的函数在指定范围内的表现。

    参数：
        func (callable): 一个接受两个参数的可调用函数，代表要可视化的函数。
        x_range (tuple): 包含两个元素的元组，表示x轴的范围（最小值和最大值）。
        y_range (tuple): 包含两个元素的元组，表示y轴的范围（最小值和最大值）。
    """
    # 设置网格点的数量
    num_points = 200
    x_start, x_end = x_range
    y_start, y_end = y_range

    # 生成x和y的一维数组
    x_vals = np.linspace(x_start, x_end, num_points)
    y_vals = np.linspace(y_start, y_end, num_points)

    # 创建网格坐标矩阵
    X, Y = np.meshgrid(x_vals, y_vals)

    # 使用向量化函数计算每个点的函数值
    vec_func = np.vectorize(func, otypes=[np.float64])
    func_values = vec_func(X.ravel(), Y.ravel()).reshape(X.shape)

    # 绘制等高线图并填充颜色
    plt.figure(figsize=(8, 6))
    CS = plt.contourf(X, Y, func_values, levels=20)
    plt.colorbar(CS, label='Function Value')

    # 添加坐标标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Contour Plot of Function {func.__name__}')

    # 显示图形
    # plt.show()


class NetP:
    def __init__(self,net:torch.nn.Module,xdir,ydir,netf):
        self.net =  net
        self.xdir = xdir
        self.ydir = ydir
        self.lr = 1e-0
        self.netf = netf

    @torch.no_grad()
    def __call__(self,x,y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        xsp,ysp = x.shape,y.shape

        xf = x.flatten()
        yf = y.flatten()
        z = {}
        for x,y,i in zip(xf,yf,range(xf.size(0))):
            # print(i,end=' ',flush=True)
            tn = self.netf()
            for t,p,xd,yd in zip(tn.parameters(),self.net.parameters(),self.xdir.parameters(),self.ydir.parameters()):
                t.copy_(p.data + xd * x * self.lr + yd * y * self.lr)
            z[i] = LOSSF(tn)
            tn.cpu()

        z = np.array([z[i] for i in range(xf.size(0))])
        z.resize(xsp)
        return z

class Visual:
    
    def __init__(self,center:Module,data,netf):

        self.center = center
        self.a_batch = None
        for _batch in DataLoader(data,1024,shuffle=False):
            self.a_batch = [_batch[0],_batch[1]]
            break
        self._lossf = LOSSF
        self.netf = netf
        self.X,self.Y = self.initDir()
        plot_contour(NetP(self.center,self.X,self.Y,netf),(-1,1),(-1,1))

    @torch.no_grad()
    def _lossf(self,net):
        o = net(self.a_batch[0])
        loss = self._loss(o,self.a_batch[1]).detach().item()
        return loss

    @torch.no_grad()
    def loc(self,target:Module):
        """
            计算target模型相对center模型的坐标。
            以self.X和self.Y为基底。
        """
        x,y,counter = 0,0,0
        for tr,so,xdir,ydir in zip(target.parameters(),self.center.parameters(),self.X.parameters(),self.Y.parameters()):
            counter += 1
            x += (tr - so) / xdir
            y += (tr - so) / ydir
        return x.item()/counter,y.item()/counter

    @torch.no_grad()
    def initDir(self,xpth = XDIR,ypth = YDIR):
        x = self.netf()
        if os.path.exists(xpth):
            x.load_state_dict(torch.load(xpth,weights_only=True))
        else:
            for xp in x.parameters():
                xp.div_(xp.norm() + 1e-4)
            torch.save(x.state_dict(),xpth)

        y = self.netf()
        if os.path.exists(ypth):
            y.load_state_dict(torch.load(ypth,weights_only=True))
        else:
            for yp,xp in zip(y.parameters(),x.parameters()):
                while (yp + xp).norm() < 1e-1:
                    yp.copy_(torch.randn_like(yp))
                    yp.div_(yp.norm() + 1e-4)
            torch.save(y.state_dict(),ypth)

        return x,y

if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.datasets.mnist import EMNIST


    class Div255(torch.nn.Module):
        def forward(self,x:torch.Tensor):
            return x.div_(255.0)

    nets = []
    center = None
    data = EMNIST( 
            root='./datasets/EMNIST/',
            train=True,
            download=False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                Div255()
            ]),
            split='byclass',
        )
    xs,ys = [],[]

    ver = Visual()
    for net in nets:
        x,y = ver.loc(net)
        xs.append(x)
        ys.append(y)
    
    plt.scatter(xs,ys)
    plt.show()