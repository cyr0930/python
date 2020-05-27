import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# https://github.com/pytorch/pytorch/issues/30966
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def basic():
    x = torch.tensor([5.5, 3])
    x = x.new_ones((5, 3), dtype=torch.double)
    y = torch.rand(5, 3)
    y += x
    print(y.view(-1, 5).size())     # Resizing

    # The Torch Tensor and NumPy array will share their underlying memory locations
    # (if the Torch Tensor is on CPU), and changing one will change the other
    a = y[: 1]
    b = a.numpy()
    print(a, b)
    val = torch.randn(1).item()     # If you have a one element tensor, use .item() to get the value as a Python number
    a.add_(val)     # in-place
    print(a, b)
    b = torch.from_numpy(b)
    print(b)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        b = b.to(device)
        print(b, b.to("cpu"))


def autograd():
    # Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete
    # history of computation. Each tensor has a .grad_fn attribute that references a Function that has
    # created the Tensor (except for Tensors created by the user - their grad_fn is None).
    x = torch.ones(2, 2, requires_grad=True)    # if requires_grad = True, it starts to track all operations on it
    print(x)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    print(out)
    print(out.grad_fn.next_functions[0][0])
    out.backward()  # all the gradients computed automatically
    print(x.grad)   # d(out)/dx

    # If Tensor isn't a scalar, you need to specify a gradient argument that is a tensor of matching shape
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)
    print(x.grad)

    # Stop autograd from tracking history on Tensors
    # either by wrapping the code block in "with torch.no_grad():" or by using .detach()
    with torch.no_grad():
        print((x ** 2).requires_grad)
    x = x.detach()
    print(x.requires_grad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # You just have to define the forward function, and the backward function
        # is automatically defined for you using autograd
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def neural_networks():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    n, bs = 12000, 2000
    data = torch.randn(n, 1, 32, 32).to(device)
    target = torch.randint(10, (n,)).to(device)  # a dummy target
    classes = list('abcdefghij')

    writer = SummaryWriter()
    data0, target0 = data[:10], target[:10]
    class_target0 = [classes[lab] for lab in target0]
    features = data0.view(-1, 32 * 32)
    writer.add_graph(net, data0)
    writer.add_embedding(features, metadata=class_target0)

    for epoch in range(4):
        running_loss = 0.0
        for i in range(0, n, bs):
            inputs, labels = data[i:i+bs], target[i:i+bs]
            optimizer.zero_grad()   # clear the existing gradients though, else gradients will be accumulated
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    # does the update

            running_loss += loss.item()
            if i % bs == 0:
                global_step = epoch*n+i+bs
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.data, global_step)
                    writer.add_histogram(name+'_grad', param.grad, global_step)
                writer.add_scalar('training loss', running_loss/bs, global_step)
                running_loss = 0.0

    print('conv1.bias.grad', net.conv1.bias.grad)
    print('Finished Training')
    writer.close()

    PATH = 'tmp/cnn.pth'
    torch.save(net.state_dict(), PATH)
    net = Net()
    net.load_state_dict(torch.load(PATH))


if __name__ == "__main__":
    # basic()
    # autograd()
    neural_networks()
