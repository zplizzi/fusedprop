import torch
from torch import nn
import torch.autograd as autograd
import torch.nn.functional as F


def reduce(x, reduction=None):
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


def generator_loss(dgz, args):
    """Expects raw outputs without sigmoid."""
    if args.loss == "minimax":
        return minimax_generator_loss(dgz, nonsaturating=False)
    elif args.loss == "nonsaturating":
        return minimax_generator_loss(dgz, nonsaturating=True)
    elif args.loss == "least_squares":
        return least_squares_generator_loss(dgz)
    elif args.loss == "wasserstein":
        return wasserstein_generator_loss(dgz)
    elif args.loss == "hinge":
        return hinge_generator_loss(dgz)
    else:
        raise ValueError


def discriminator_loss(dx, dgz, args):
    """Expects raw outputs without sigmoid."""
    if args.loss == "minimax":
        return minimax_discriminator_loss(dx, dgz)
    elif args.loss == "nonsaturating":
        return minimax_discriminator_loss(dx, dgz)
    elif args.loss == "least_squares":
        return least_squares_discriminator_loss(dx, dgz)
    elif args.loss == "wasserstein":
        return wasserstein_discriminator_loss(dx, dgz)
    elif args.loss == "hinge":
        return hinge_discriminator_loss(dx, dgz)
    else:
        raise ValueError


def lambda_inv_loss(dx, dgz, args):
    """Compute loss for the gradient-reversal path where we're using a
    lambda-inverse term.
    """
    # This should be the normal discriminator loss for the real samples
    # But for fake samples, it should be the normal generator loss.

    if args.loss == "minimax":
        raise ValueError
    elif args.loss == "nonsaturating":
        real_loss = F.softplus(-dx)
        fake_loss = minimax_generator_loss(dgz, nonsaturating=True)
    elif args.loss == "least_squares":
        raise ValueError
    elif args.loss == "wasserstein":
        raise ValueError
    elif args.loss == "hinge":
        real_loss = nn.functional.relu(-dx + 1)
        fake_loss = hinge_generator_loss(dgz)

    return reduce(real_loss + fake_loss, "mean")


def grad_reversal_lambda(dgz, args):
    """Expects raw outputs without sigmoid."""
    if args.loss == "minimax":
        return -1 * torch.ones_like(dgz)
    elif args.loss == "nonsaturating":
        return -1 * torch.exp(-1 * dgz)
    elif args.loss == "least_squares":
        eps = 1e-5
        return (dgz - 1) / (dgz + eps)
    elif args.loss == "wasserstein":
        return -1 * torch.ones_like(dgz)
    else:
        raise ValueError


def grad_reversal_lambda_inv(dgz, args):
    """Expects raw outputs without sigmoid."""
    if args.loss == "minimax":
        raise ValueError
    elif args.loss == "nonsaturating":
        return -1 * torch.exp(dgz)
    elif args.loss == "least_squares":
        raise ValueError
    elif args.loss == "wasserstein":
        raise ValueError
    elif args.loss == "hinge":
        return -1 * ((dgz + 1) > 0).int().float()
    else:
        raise ValueError

# Minimax


def minimax_generator_loss(dgz, nonsaturating=True, reduction="mean"):
    if nonsaturating:
        return reduce(F.softplus(-dgz), reduction)
    else:
        return reduce(-F.softplus(dgz), reduction)


def minimax_discriminator_loss(dx, dgz, label_smoothing=0.0, reduction="mean"):
    loss_real = F.softplus(-dx)
    loss_fake = F.softplus(dgz)
    loss = loss_real + loss_fake
    loss = reduce(loss, reduction)
    return loss


# Hinge


def hinge_generator_loss(dgz, reduction="mean"):
    return reduce(-dgz, reduction)


def hinge_discriminator_loss(dx, dgz, reduction="mean"):
    real_loss = nn.functional.relu(-dx + 1)
    fake_loss = nn.functional.relu(dgz + 1)
    return reduce(real_loss + fake_loss, reduction)



# Least Squared Losses


def least_squares_generator_loss(dgz, c=1.0, reduction="mean"):
    return 0.5 * reduce((dgz - c)**2, reduction)


def least_squares_discriminator_loss(dx, dgz, a=0.0, b=1.0, reduction="mean"):
    return 0.5 * (reduce((dx - b)**2, reduction) + reduce(
        (dgz - a)**2, reduction))


# Wasserstein


def wasserstein_generator_loss(fgz, reduction="mean"):
    return reduce(-1.0 * fgz, reduction)


def wasserstein_discriminator_loss(fx, fgz, reduction="mean"):
    return reduce(fgz - fx, reduction)


def wasserstein_gradient_penalty(interpolate, d_interpolate, reduction="mean"):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Implementation in "torchgan"
    # gradient_penalty = (gradients.norm(2) - 1)**2
    # return reduce(gradient_penalty, reduction)

    # Implementation in "comparegan". Not sure the point of the .0001
    # The careful reduction aims to get the slope for each training sample,
    # which should be close to 1.
    slopes = torch.sqrt(gradients.pow(2).sum(dim=(1, 2, 3)) + .0001)
    gradient_penalty = (slopes - 1.0).pow(2).mean()
    return gradient_penalty
