from torch import nn, optim, from_numpy, cuda
from ml_models.logistic_regression import LogisticRegression
from ml_models.simple_nn import SimpleNN

criterion = nn.BCELoss(reduction = 'mean')

def train_model(x_data, y_data):
    print(cuda.is_available())
    epoch_count = 20000
    learning_rate = 0.01

    x_data = x_data.float()
    y_data = y_data.float()
    model = SimpleNN(80*60*3, 20, 6, 1)
    #model = LogisticRegression(80*60*3, 1)
    #criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_count):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_data)

        # Compute and print loss
        loss = criterion(y_pred, y_data)
        if(epoch % 99 == 0):
            print(f'Epoch: {epoch + 1}/epoch_count | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def train_model_ext(x_data, y_data, x_data_dev, y_data_dev):
    print(cuda.is_available())
    epoch_count = 20000
    learning_rate = 0.01

    x_data = x_data.float()
    y_data = y_data.float()

    x_data_dev = x_data_dev.float()
    y_data_dev = y_data_dev.float()

    model = SimpleNN(x_data.shape[1], 20, 6, 1)
    #model = LogisticRegression(80*60*3, 1)
    #criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_count):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_data)
        y_pred_dev = model(x_data_dev)

        # Compute and print loss
        loss = criterion(y_pred, y_data)
        loss_dev = criterion(y_pred_dev, y_data_dev)
        if((epoch + 1) % 100 == 0):
            print(f'Epoch: {epoch + 1}/{epoch_count} | Loss: {loss.item():.4f} | Dev loss: {loss_dev.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def check_model(x_data, y_data, model):
    x_data = x_data.float()
    y_data = y_data.float()

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Model checking | Loss: {loss.item():.4f}')
