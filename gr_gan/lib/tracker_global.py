import os
import time
import math

import torch
import wandb

i = 0


def increment_i():
    global i
    i += 1


config = None


def init(name=None, args=None, project="test", group=None):
    print(name)
    global i
    global last_time
    global run_id
    global project_id
    global config
    config = args
    last_time = None
    project_id = project
    print(args)
    # Needed to fix a bug in wandb
    wandb.watch_called = False
    wandb.init(project=project,
               config=args,
               name=name,
               resume=name,
               group=group,
               dir="/tmp/wandb/")


def add_histogram(tag, data, freq=100, plot_mean=False):
    if type(data) == torch.Tensor:
        data = data.cpu().detach()
    wandb.log({tag: wandb.Histogram(data)}, step=i)
    if plot_mean:
        add_scalar(tag + "_mean", data.mean(), freq=1)


def add_chart(tag, chart, freq=100):
    wandb.log({tag: chart}, step=i)


def add_scalar(tag, value, freq=20):
    if i % freq != 0:
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: value}, step=i)


def add_image(tag, value, caption="label", freq=100):
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: [wandb.Image(value, caption=caption)]}, step=i)


def add_image_grid(tag, x, freq=100, range=None):
    shape = x.shape
    if len(shape) == 3:
        # Greyscale inputs like (batch_size, dim, dim)
        x = x.reshape((-1, 1, x.shape[1], x.shape[2]))
    elif len(shape) == 4:
        # Color inputs like (batch_size, 3, dim, dim)
        pass
    else:
        raise ValueError
    grid = make_grid(x, normalize=True, scale_each=True, range=range)
    add_image(tag, grid, freq=1)


def watch(model, freq=50):
    wandb.watch(model, log="all", log_freq=freq)


last_time = None


def log_iteration_time(batch_size, freq=10):
    """Call this once per training iteration."""
    global last_time
    if i % freq != 0:
        return

    if last_time is None:
        last_time = time.time()
    else:
        dt = (time.time() - last_time) / freq
        last_time = time.time()
        add_scalar("timings/iterations-per-sec", 1 / dt, freq=1)
        add_scalar("timings/samples-per-sec", batch_size / dt, freq=1)


def checkpoint_model(model, freq, model_name="model"):
    if i > 0 and i % freq == 0:
        print(f"beginning checkpoint of {model_name}")
        filename = f"{model_name}.{i}.pytorch"
        # Write model to local dir
        path = os.path.join(wandb.run.dir, filename)
        # This probably won't work not in dataparallel
        try:
            torch.save(model.module.state_dict(), path)
        except AttributeError:
            torch.save(model.state_dict(), path)
        # Send up to cloud
        wandb.save(filename)
        wandb.run.summary["checkpoint_i"] = i
        print(f"finished checkpoint of {model_name}")


irange = range

# This is from torchvision.utils.make_grid but has a bug that is yet to be fixed in pip


def make_grid(tensor,
              nrow=8,
              padding=2,
              normalize=False,
              range=None,
              scale_each=False,
              pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(
            type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) +
                        padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding, width * xmaps + padding),
        pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid
