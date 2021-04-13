import torch
import torch.nn.functional as F
import numpy as np


class Model(torch.nn.Module):
    def __init__(self,grad):
        super(Model,self).__init__()
        self.grad= grad
    def forward(self,x):
        x = x @ self.grad['W1']

        zeros = torch.tensor(np.zeros(x.shape))
        x = torch.maximum(zeros,x)
        
        x = x @ self.grad['W2']
        x = F.relu(x)
        
        x = torch.sum(torch.square(x),dim=1)
        return x
    
grad = {}
grad['W1'] = torch.FloatTensor(np.random.rand(2,2))
grad['W2'] = torch.DoubleTensor(np.random.rand(2,1))
dout = torch.FloatTensor(np.random.rand(2))

model = Model(grad)
x = torch.FloatTensor(np.random.rand(2,2))
x.requires_grad_(True)

out = model(x)

print(out)
out.backward(dout)
print(x.grad)
