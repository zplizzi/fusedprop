import os

import torch
from imageio import imsave
from tqdm import tqdm

from gr_gan.lib.tf_fid_score import calculate_fid_given_paths, check_or_download_inception, create_inception_graph
from gr_gan.lib.tf_inception_score import get_inception_score, _init_inception

import tempfile


# initialize tensorflow stuff:
_init_inception()
inception_path = check_or_download_inception(None)
create_inception_graph(inception_path)


def validate(G, args):
    fid_stat_path = os.path.join(args.data_root,
                                 f"fid_stat/fid_stats_{args.dataset}_train.npz")

    G.eval()

    with tempfile.TemporaryDirectory() as fid_buffer_dir:
        # get fid and inception score

        # TODO: it's not clear to me how many samples were used in the
        # pre-computed FID stats.
        assert args.fid_real_samples == args.fid_fake_samples
        eval_iter = args.fid_real_samples // args.batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.randn((args.batch_size, args.z_dim)).to(args.device)

            # Generate a batch of images
            gen_imgs = G(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
                imsave(file_name, img)
            img_list.extend(list(gen_imgs))

        # get inception score
        is_mean, is_std = get_inception_score(img_list)

        # get fid score
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat_path], inception_path=None)

    G.train()

    return is_mean, fid_score
