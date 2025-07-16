import torch
import torch.nn as nn
import torch.optim as optim
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

class TwoLayerNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # return self.fc1(x)
        # this fails:
        x = torch.relu(self.fc1(x))
        # this works:
        # x = self.fc1(x)
        # perhaps the problem is that the intermediate activation is not being captured correctly...
        return self.fc2(x)

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        torch.manual_seed(0)
        self.model = TwoLayerNet().to(self.device)
        self.example_inputs = torch.randn(self.batch_size, 100, device=self.device)
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        if self.test == "train":
            self.model.train()
        else:
            self.model.eval()

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        self.optimizer.zero_grad()
        output = self.model(self.example_inputs)
        loss = self.loss_fn(output, self.example_target)
        loss.backward()
        self.optimizer.step()

    def eval(self):
        with torch.no_grad():
            out = self.model(self.example_inputs)
        return (out,)
