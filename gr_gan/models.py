import torch
from torch import nn
from torch.nn import functional as F

from gr_gan import layers
from gr_gan import models_resnet


def get_models(args):
    if args.model == "mnist_fc":
        G = Generator_MNIST_FC(args)
        D = Discriminator_MNIST_FC(args)
    elif args.model == "sngan":
        G = Generator_SNGAN(args)
        D = Discriminator_SNGAN(args)
    elif args.model == "infogan":
        G = Generator_INFOGAN(args)
        D = Discriminator_INFOGAN(args)
    elif args.model == "resnet":
        G = models_resnet.ResNetGenerator(args)
        D = models_resnet.ResNetDiscriminator(args)
    else:
        raise ValueError

    return G.to(args.device), D.to(args.device)


class Generator_MNIST_FC(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.z_dim != 128:
            print("not using recommended z dim of 100!")
        self.fc1 = nn.Linear(args.z_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 28 * 28)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        x = x.reshape((batch_size, 1, 28, 28))
        return x


class Discriminator_MNIST_FC(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.spectral_norm:
            sn = torch.nn.utils.spectral_norm
        else:
            sn = lambda x: x

        Linear = layers.GradScalingLinear

        self.fc1 = sn(Linear(28 * 28, 1024))
        self.fc2 = sn(
            Linear(self.fc1.out_features, self.fc1.out_features // 2))
        self.fc3 = sn(
            Linear(self.fc2.out_features, self.fc2.out_features // 2))
        self.fc4 = sn(Linear(self.fc3.out_features, 1))
        self.dropout = nn.Dropout(.3)

    # forward method
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, 28 * 28))
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class Generator_SNGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.z_dim != 128:
            print("not using recommended z dim of 128!")
        self.fc1 = nn.Linear(args.z_dim, 512 * (args.dim // 8)**2)

        self.bn1 = nn.BatchNorm1d(512 * (args.dim // 8)**2)
        self.c1 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.c2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.c3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.c4 = nn.ConvTranspose2d(
            64,
            args.color_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)

        torch.nn.init.normal_(self.fc1.weight, std=.02)
        torch.nn.init.normal_(self.c1.weight, std=.02)
        torch.nn.init.normal_(self.c2.weight, std=.02)
        torch.nn.init.normal_(self.c3.weight, std=.02)
        torch.nn.init.normal_(self.c4.weight, std=.02)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.init.zeros_(self.c3.bias)
        torch.nn.init.zeros_(self.c4.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.args.z_dim)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = x.reshape((batch_size, 512, self.args.dim // 8,
                       self.args.dim // 8))

        x = F.relu(self.bn2(self.c1(x)))
        x = F.relu(self.bn3(self.c2(x)))
        x = F.relu(self.bn4(self.c3(x)))

        x = self.c4(x)
        x = torch.tanh(x)
        assert x.shape == (batch_size, self.args.color_channels, self.args.dim,
                           self.args.dim)
        return x


class Discriminator_SNGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.spectral_norm:
            sn = torch.nn.utils.spectral_norm
        else:
            sn = lambda x: x

        if args.lambda_inv:
            Linear = layers.GradScalingLinear
            Conv2d = layers.GradScalingConv2d
        else:
            Linear = nn.Linear
            Conv2d = nn.Conv2d

        self.c1 = sn(
            Conv2d(
                args.color_channels, 64, kernel_size=3, stride=1, padding=1))
        self.c2 = sn(Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.c3 = sn(Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.c4 = sn(Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.c5 = sn(Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.c6 = sn(Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.c7 = sn(Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.fc1 = sn(Linear(512 * (args.dim // 8)**2, 1))

        torch.nn.init.normal_(self.fc1.weight, std=.02)
        torch.nn.init.normal_(self.c1.weight, std=.02)
        torch.nn.init.normal_(self.c2.weight, std=.02)
        torch.nn.init.normal_(self.c3.weight, std=.02)
        torch.nn.init.normal_(self.c4.weight, std=.02)
        torch.nn.init.normal_(self.c5.weight, std=.02)
        torch.nn.init.normal_(self.c6.weight, std=.02)
        torch.nn.init.normal_(self.c7.weight, std=.02)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.init.zeros_(self.c3.bias)
        torch.nn.init.zeros_(self.c4.bias)
        torch.nn.init.zeros_(self.c5.bias)
        torch.nn.init.zeros_(self.c6.bias)
        torch.nn.init.zeros_(self.c7.bias)

    def forward(self, x, dropout_p=0.0):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.args.color_channels, self.args.dim,
                           self.args.dim)

        x = F.leaky_relu(self.c1(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c2(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c3(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c4(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c5(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c6(x), 0.1)
        x = torch.nn.functional.dropout2d(x, p=dropout_p)
        x = F.leaky_relu(self.c7(x), 0.1)

        assert x.shape == (batch_size, 512, self.args.dim // 8,
                           self.args.dim // 8), x.shape
        x = x.reshape((batch_size, 512 * (self.args.dim // 8)**2))
        x = self.fc1(x)
        assert x.shape == (batch_size, 1)
        return x


class Generator_INFOGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = args.dim
        if args.z_dim != 64:
            print("not using recommended z dim of 64!")
        self.fc1 = nn.Linear(args.z_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128 * (dim // 4)**2)
        self.bn2 = nn.BatchNorm1d(128 * (dim // 4)**2)

        self.c1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.c2 = nn.ConvTranspose2d(
            64,
            args.color_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.args.z_dim)
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = x.reshape((batch_size, 128, self.args.dim // 4,
                       self.args.dim // 4))
        x = F.leaky_relu(self.bn3(self.c1(x)), 0.2)
        x = self.c2(x)

        x = torch.tanh(x)
        assert x.shape == (batch_size, self.args.color_channels, self.args.dim,
                           self.args.dim)
        return x


class Discriminator_INFOGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.spectral_norm:
            sn = torch.nn.utils.spectral_norm
        else:
            sn = lambda x: x

        self.c1 = sn(
            nn.Conv2d(
                args.color_channels, 64, kernel_size=4, stride=2, padding=1))
        self.c2 = sn(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(128)
        self.fc1 = sn(nn.Linear(128 * (args.dim // 4)**2, 1024))
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = sn(nn.Linear(1024, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.args.color_channels, self.args.dim,
                           self.args.dim)

        # intentionally not including bn after c1, matching comparegan
        x = self.c1(x)
        x = F.leaky_relu(x, 0.2)

        # Again, bn before relu
        x = self.c2(x)
        if self.args.batch_norm:
            x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = x.reshape((batch_size, 128 * ((self.args.dim // 4)**2)))

        x = self.fc1(x)
        if self.args.batch_norm:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc2(x)
        assert x.shape == (batch_size, 1)
        return x
