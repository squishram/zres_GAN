import torch

# Here we replace the manually computed gradient with autograd


# model output
def forward(x):
    return w * x


# cost = MSE
def cost(y, y_pred):
    return ((y_pred - y)**2).mean()


# Linear regression
# f = w * x

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # cost
    loss = cost(Y, y_pred)

    # calculate gradients = backward pass
    loss.backward()

    # update weights
    # w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero the gradients after updating
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, cost = {loss.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

