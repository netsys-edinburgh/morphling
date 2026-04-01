import torch
from torch import nn
from torch.optim import SGD


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def test_optimizer_write(tmp_path):
    model = Net()
    optimizer = SGD(model.parameters(), lr=0.1)
    torch.save(optimizer, tmp_path / "optimizer.pt")

    optimizer.zero_grad()
    output = model(torch.randn(1, 1))
    output.backward()
    optimizer.step()
