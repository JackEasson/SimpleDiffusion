import os
import argparse
import torch
from yacs.config import CfgNode as CN
from models.simpleunet import GhostEGEUnet
from torchvision.utils import save_image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--exp_dir', type=str, default='result',
        help='Path to the result directory.',
    )
    parser.add_argument(
        '-m', '--ckpt_path', type=str, default='outputs/exp-2023-07-30-23-16-58/ckpt/ckpt_10.pt',
        help='Path to the checkpoints',
    )
    args = parser.parse_args()
    return args


def get_config():
    cfg = CN(new_allowed=True)
    """训练参数设置"""
    cfg.device = 'cuda:0'

    """模型参数设置"""
    cfg.model = CN(new_allowed=True)
    cfg.model.image_size = 128
    cfg.model.T = 1000
    cfg.model.temb_chn = 256
    cfg.model.c_list = [16, 24, 32, 48, 64, 96]
    cfg.model.width = 1.5
    cfg.model.final_activate = 'tanh'
    cfg.model.bridge = True

    cfg.freeze()
    return cfg


def train(args, cfg):
    # PREPARE
    exp_dir = args.exp_dir
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    device = cfg.device
    model = GhostEGEUnet(input_channels=3,
                         output_channels=3,
                         T=cfg.model.T,
                         temb_channels=cfg.model.temb_chn,
                         c_list=cfg.model.c_list,
                         width=cfg.model.width,
                         final_activate=cfg.model.final_activate,
                         bridge=cfg.model.bridge).to(device)

    ckpt_model = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt_model)
    print(f'Successfully load model from {args.ckpt_path}')

    model.eval()
    beta_1 = 1e-4
    beta_T = 0.02
    T = cfg.model.T
    betas = torch.linspace(beta_1, beta_T, T).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    x_T = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size).to(device)  # 采样自标准正态分布的x_T
    x_t = x_T
    with torch.no_grad():
        for t_step in reversed(range(T)):  # 从T开始向零迭代
            t = t_step
            t = torch.tensor(t).to(device)
            print(t, t.reshape(1, ))
            z = torch.randn_like(x_t, device=device) if t_step > 0 else 0  # 如果t大于零，则采样自标准正态分布，否则为零
            """按照公式计算x_{t-1}"""
            x_t_minus_one = torch.sqrt(1 / alphas[t]) * (
                    x_t - (1 - alphas[t]) * model(x_t, t.reshape(1, )) / torch.sqrt(1 - alphas_bar[t])) + torch.sqrt(
                betas[t]) * z

            x_t = x_t_minus_one
            print(t_step)
        # x_0 = torch.clip(x_t,-1,1)

        x_0 = x_t
        x_0 = x_0 * 0.5 + 0.5
        save_image(x_0, 'sample.jpg')

def main():
    args = get_parser()
    cfg = get_config()
    train(args, cfg)


if __name__ == '__main__':
    main()
