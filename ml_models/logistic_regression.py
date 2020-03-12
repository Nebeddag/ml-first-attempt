import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
     def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

def train_model(x_data: Variable, y_data: Variable):
    model = LogisticRegression
    criterion = torch.nn.BCELoss(size_average = True)
    optimizer = torch.optim.SGD(model.parameters, lr = 0.01)

    for epoch in range(1000):
        y_pred = model(x_data)
        
        loss = criterion(y_pred, y_data)
        print(epoch, loss.data)

        optimizer.zero.grad()
        loss.backward()
        optimizer.step()

