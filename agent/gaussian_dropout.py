import torch
import torch.nn as nn

class GaussianDropout(nn.Module):
    def __init__(self, p=0.2):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
        
    def forward(self, x, execute_dropout):
        if execute_dropout: # when trainning 
            stddev = (self.p / (1.0 - self.p))**0.5 # 标准差
            epsilon = torch.randn_like(x) * stddev + 1 # 均值为0，方差为1的正太分布，尺寸与x相同
            return x * epsilon
        else:
            return x