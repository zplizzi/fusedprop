import torch

from gr_gan import models
from gr_gan import datasets
from gr_gan import losses
from gr_gan import layers
from gr_gan import tf_validation
from gr_gan.lib import tracker_global as t


class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cudnn.benchmark = True

        t.init(
            name=self.args.name,
            project=self.args.project,
            args=self.args,
            group=self.args.group)

        self.train_loader, self.test_loader = datasets.get_dataset(self.args)
        self.dataloader_iterator = iter(self.train_loader)

        self.G, self.D = models.get_models(self.args)

        if not self.args.lr_dis:
            self.args.lr_dis = self.args.lr

        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2))
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=self.args.lr_dis,
            betas=(self.args.beta1, self.args.beta2))

        t.watch((self.G, self.D), freq=self.args.log_freq * 100)

    def train(self):
        while True:
            if self.args.iterations and t.i > self.args.iterations:
                break

            self.train_step()

            if t.i % self.args.evaluate_freq == 0:
                tf_is, tf_fid = tf_validation.validate(self.G, self.args)
                t.add_scalar("TF_FID", tf_fid, freq=1)
                t.add_scalar("TF_IS", tf_is, freq=1)

    def get_batch(self):
        try:
            batch = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.train_loader)
            batch = next(self.dataloader_iterator)
        return batch

    def train_step(self):
        batch = self.get_batch()
        if self.args.train_fn == "reversal":
            self._train_step_reversal(batch, logging=True)
        elif self.args.train_fn == "reversal_sim":
            self._train_step_reversal_sim(batch, logging=True)
        elif self.args.train_fn == "baseline":
            self._train_step_original(batch, logging=True)
        else:
            raise ValueError

        t.log_iteration_time(self.args.batch_size, freq=20)
        t.increment_i()
        t.checkpoint_model(self.G, freq=20000, model_name="generator")
        t.checkpoint_model(self.D, freq=20000, model_name="discriminator")

    def _train_step_original(self, batch, logging):
        freq = self.args.log_freq

        # Train Generator
        self.G.zero_grad()
        z = torch.randn((self.args.batch_size, self.args.z_dim),
                        device=self.args.device)
        x_fake = self.G(z)
        Dx_fake = self.D(x_fake)
        G_loss = losses.generator_loss(Dx_fake, self.args)

        G_loss.backward()
        if not self.args.sim_gd:
            self.opt_G.step()

        # Train Discriminator
        for i in range(self.args.n_dis):
            # Get a new batch so we can do a loop here
            batch = self.get_batch()
            x_real, labels = batch
            x_real = x_real.to(self.args.device)
            labels = labels.to(self.args.device)

            # Train Discriminator
            self.D.zero_grad()
            z = torch.randn((len(x_real), self.args.z_dim),
                            device=self.args.device)
            x_fake = self.G(z).detach()

            x_combined = torch.cat((x_real, x_fake), dim=0)
            Dx_combined = self.D(x_combined)
            Dx_real = Dx_combined[:len(x_real)]
            Dx_fake = Dx_combined[len(x_real):]

            D_loss = losses.discriminator_loss(Dx_real, Dx_fake, self.args)
            loss = D_loss

            loss.backward()

            self.opt_D.step()

        if self.args.sim_gd:
            assert self.args.n_dis == 1
            self.opt_G.step()

        if logging and t.i % freq == 0:
            t.add_scalar("d loss", D_loss, freq=1)
            t.add_scalar("g loss", G_loss, freq=1)

        if logging and t.i % (freq * 10) == 0:
            t.add_histogram("debug/D_real", Dx_real, freq=1, plot_mean=True)
            t.add_histogram("debug/D_fake", Dx_fake, freq=1, plot_mean=True)

        if logging and t.i % (freq * 100) == 0:
            t.add_image_grid("generated", x_fake[:20], freq=1)
            t.add_image_grid("real", x_real[:20], freq=1)

    def _train_step_reversal(self, batch, logging):
        x_real, labels = batch
        x_real = x_real.to(self.args.device)
        labels = labels.to(self.args.device)

        self.opt_G.zero_grad()
        self.opt_D.zero_grad()

        z = torch.randn((len(x_real), self.args.z_dim),
                        device=self.args.device)
        x_fake = self.G(z)
        if not self.args.lambda_inv:
            x_fake = layers.grad_scaling(x_fake)

        x_combined = torch.cat((x_real, x_fake), dim=0)
        Dx_combined = self.D(x_combined)
        Dx_real = Dx_combined[:len(x_real)]
        Dx_fake = Dx_combined[len(x_real):]

        if not self.args.lambda_inv:
            # We just use the normal discriminator loss
            D_loss = losses.discriminator_loss(Dx_real, Dx_fake, self.args)

            # Compute grad reversal lambda - which will be used on the .backward()
            # call.
            layers.scale1 = losses.grad_reversal_lambda(Dx_fake, self.args)

        else:
            D_loss = losses.lambda_inv_loss(Dx_real, Dx_fake, self.args)

            # We don't scale the generator in inverse mode
            layers.scale1 = torch.ones(
                len(x_real), device=self.args.device)

            # We don't scale the grad for real samples
            real_scale = torch.ones(len(x_real), device=self.args.device)
            fake_scale = losses.grad_reversal_lambda_inv(Dx_fake,
                                                         self.args).flatten()
            layers.scale2 = torch.cat((real_scale, fake_scale), dim=0)

        D_loss.backward()

        freq = self.args.log_freq

        self.opt_G.step()
        self.opt_D.step()

        if logging and t.i % freq == 0:
            if not self.args.lambda_inv:
                t.add_scalar("d loss", D_loss, freq=1)
            else:
                # Simulate D loss
                D_loss = losses.discriminator_loss(Dx_real, Dx_fake, self.args)
                t.add_scalar("d loss", D_loss, freq=1)

            # Simulate G loss
            G_loss = losses.generator_loss(Dx_fake, self.args)
            t.add_scalar("g loss", G_loss, freq=1)

        if logging and t.i % (freq * 10) == 0:
            generated = x_fake
            t.add_histogram("debug/D_real", Dx_real, freq=1, plot_mean=True)
            t.add_histogram("debug/D_fake", Dx_fake, freq=1, plot_mean=True)

        if logging and t.i % (freq * 100) == 0:
            t.add_image_grid("generated", generated[:20], freq=1)
            t.add_image_grid("real", x_real[:20], freq=1)
