from torch import nn, optim, from_numpy
from ml_models.logistic_regression import LogisticRegression
from ml_models.simple_nn import SimpleNN


def train_model(x_data, y_data):
    epoch_count = 10000
    learning_rate = 0.02

    x_data = x_data.float()
    y_data = y_data.float()
    model = SimpleNN(80*60*3, 20, 6, 1)
    #model = LogisticRegression(80*60*3, 1)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    for epoch in range(epoch_count):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_data)

        # Compute and print loss
        loss = criterion(y_pred, y_data)
        if(epoch % 100 == 0):
            print(f'Epoch: {epoch + 1}/epoch_count | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
