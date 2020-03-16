from torch import nn, optim, from_numpy


class SimpleNN(nn.Module):
    def __init__(self, inp_dim, h1_dim, h2_dim, out_dim):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(SimpleNN, self).__init__()
        self.l1 = nn.Linear(inp_dim, h1_dim)
        self.l2 = nn.Linear(h1_dim, h2_dim)
        self.l3 = nn.Linear(h2_dim, out_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
