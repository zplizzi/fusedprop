import torch
from torchvision import transforms, datasets

import os


def make_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=6,
                                       drop_last=False,
                                       pin_memory=True)


def get_dataset(args):
    if args.dataset == "mnist":
        args.dim = 28
        args.color_channels = 1
        args.fid_real_samples = 10000
        args.fid_fake_samples = 10000
        train_dataset, test_dataset = torchvision_dataset(datasets.MNIST, args)
    elif args.dataset == "fashion_mnist":
        args.dim = 28
        args.color_channels = 1
        args.fid_real_samples = 10000
        args.fid_fake_samples = 10000
        train_dataset, test_dataset = torchvision_dataset(
            datasets.FashionMNIST, args)
    elif args.dataset == "fashion_mnist_32":
        args.dim = 32
        args.color_channels = 1
        args.fid_real_samples = 10000
        args.fid_fake_samples = 10000
        train_dataset, test_dataset = torchvision_dataset(
            datasets.FashionMNIST, args, resize=32)
    elif args.dataset == "cifar10":
        args.dim = 32
        args.color_channels = 3
        # These are the settings used in the SNGAN paper
        # args.fid_real_samples = 10000
        # args.fid_fake_samples = 5000
        # These are the settings used in CompareGan and the second google paper
        args.fid_real_samples = 10000
        args.fid_fake_samples = 10000
        train_dataset, test_dataset = torchvision_dataset(
            datasets.CIFAR10, args)
    elif args.dataset == "cifar10_gs":
        args.dim = 32
        args.color_channels = 1
        args.fid_real_samples = 10000
        args.fid_fake_samples = 10000
        train_dataset, test_dataset = torchvision_dataset(
            datasets.CIFAR10, args, grayscale=True)
    else:
        raise ValueError(args.dataset)

    print(args)

    return (make_dataloader(train_dataset, args.batch_size),
            make_dataloader(test_dataset, args.batch_size))


def torchvision_dataset(dataset, args, resize=None, grayscale=False):
    ts = []
    if resize:
        ts.append(transforms.Resize(resize))
    if grayscale:
        ts.append(transforms.Grayscale())
    ts += [
        transforms.ToTensor(),
        # Convert images to be in [-1, 1]
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ]

    transform = transforms.Compose(ts)

    data_dir = os.path.join(args.data_root, "torchvision_datasets/")
    train_dataset = dataset(root=data_dir,
                            train=True,
                            transform=transform,
                            download=True)
    test_dataset = dataset(root=data_dir,
                           train=False,
                           transform=transform,
                           download=True)

    return train_dataset, test_dataset
