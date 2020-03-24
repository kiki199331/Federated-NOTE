# Dependencies
import torch as th
import torch.nn as nn
import torch.nn.functional as F


use_cuda = th.cuda.is_available()
th.manual_seed(1)
device = th.device("cuda" if use_cuda else "cpu")

import syft as sy
from syft import WebsocketClientWorker

hook = sy.TorchHook(th)  # hook torch as always :)

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
model = Net()

# The data itself doesn't matter as long as the shape is right
mock_data = th.zeros(1, 2)

# Create a jit version of the model
traced_model = th.jit.trace(model, mock_data)


# Loss function
@th.jit.script
def loss_fn(target, pred):
    return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

# Optimizer
optimizer = "SGD"

# General hyper parameters and training options
batch_size = 4
optimizer_args = {"lr" : 0.1, "weight_decay" : 0.01}
epochs = 1
max_nr_batches = -1  # not used in this example
shuffle = True

# Create a TrainConfig

train_config = sy.TrainConfig(model=traced_model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              optimizer_args=optimizer_args,
                              epochs=epochs,
                              shuffle=shuffle)


# Connect to remote worker
kwargs_websocket = {"host": "0.0.0.0", "hook": hook, "verbose": False}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)

# Send train config
train_config.send(alice)



# evaluate our model before training.
# Setup toy data (xor example)
data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)

print("\nEvaluation before training")
pred = model(data)
loss = loss_fn(target=target, pred=pred)
print("Loss: {}".format(loss))
print("Target: {}".format(target))
print("Pred: {}".format(pred))


for epoch in range(10):
    loss = alice.fit(dataset_key="xor")  # ask alice to train using "xor" dataset
    print("-" * 50)
    print("Iteration %s: alice's loss: %s" % (epoch, loss))


new_model = train_config.model_ptr.get()

print("\nEvaluation after training:")
pred = new_model(data)
loss = loss_fn(target=target, pred=pred)
print("Loss: {}".format(loss))
print("Target: {}".format(target))
print("Pred: {}".format(pred))





