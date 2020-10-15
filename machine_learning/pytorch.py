import requests
import pickle
import gzip
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True   # may be slower
torch.backends.cudnn.benchmark = False
np.random.seed(0)


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


def neural_networks():
    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)  # cache arbitrary objects for use in the backward pass
            return input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input

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
            x = self.pool(MyReLU.apply(self.conv1(x)))
            x = self.pool(MyReLU.apply(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

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
            with autocast():
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


def mnist():
    class MnistLogistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
            self.bias = nn.Parameter(torch.zeros(10))

        def forward(self, xb):
            return xb @ self.weights + self.bias  # the @ stands for the dot product operation

    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                Lambda(lambda x: x.view(x.size(0), -1)),
            )

        def forward(self, xb):
            return self.seq(xb)

    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            batches = iter(self.dl)
            for b in batches:
                yield self.func(*b)

    DATA_PATH = Path("tmp")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    train_ds, valid_ds = TensorDataset(x_train, y_train), TensorDataset(x_valid, y_valid)

    def get_data(train_ds, valid_ds, bs):
        return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=bs * 2)

    def get_model():
        # model = MnistLogistic()
        model = MnistCNN()
        return model, optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    def loss_batch(model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)

    def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                loss_batch(model, loss_func, xb, yb, opt)
            model.eval()
            with torch.no_grad():
                losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print(epoch, val_loss / len(valid_dl))

    def preprocess(x, y):
        return x.view(-1, 1, 28, 28).to(device), y.to(device)

    bs, lr, epochs = 64, 0.1, 2
    loss_func = F.cross_entropy

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    train_dl, valid_dl = WrappedDataLoader(train_dl, preprocess), WrappedDataLoader(valid_dl, preprocess)
    model, opt = get_model()
    model.to(device)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)


if __name__ == "__main__":
    # basic()
    # autograd()
    # neural_networks()
    mnist()
