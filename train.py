import os
import argparse
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
import time
from utils.logger import StatusTracker, get_logger
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint
from dataset import LmdbDataset
from models.simpleunet import GhostEGEUnet
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--exp_dir', type=str, default='outputs',
        help='Path to the experiment directory.',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    args = parser.parse_args()
    return args


def get_config():
    cfg = CN(new_allowed=True)
    """训练参数设置"""
    cfg.epochs = 100
    cfg.batch_size = 32
    cfg.device = 'cuda:0'
    cfg.learn_rate = 0.0001
    cfg.num_workers = 0  # 数据加载核心数
    cfg.lmdb_dir = './data/flower_lmdb'
    cfg.print_freq = 50
    cfg.resume = None
    cfg.save_freq = 10  # per epoch
    cfg.seed = 12345

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
    ckpt_dir, sample_dir = create_exp_dir(exp_dir=exp_dir,
                                          cfg_dump=cfg.dump(sort_keys=False),
                                          exist_ok=cfg.resume is not None,
                                          time_str=None,
                                          no_interaction=args.no_interaction,
                                          )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
    )

    logger.info(f'Experiment directory: {exp_dir}')

    # BUILD DATASET & DATALOADER
    dataset = LmdbDataset(cfg.model.image_size, cfg.lmdb_dir)
    data_loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    logger.info(f'Size of training set: {len(dataset)}')
    logger.info(f'Total batch size: {cfg.batch_size}')

    device = cfg.device
    model = GhostEGEUnet(input_channels=3,
                         output_channels=3,
                         T=cfg.model.T,
                         temb_channels=cfg.model.temb_chn,
                         c_list=cfg.model.c_list,
                         width=cfg.model.width,
                         final_activate=cfg.model.final_activate,
                         bridge=cfg.model.bridge).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learn_rate, weight_decay=1e-4)
    loss_function = torch.nn.MSELoss()

    # ================ train details =============
    T = cfg.model.T
    beta_1 = 1e-4
    beta_T = 0.02
    betas = torch.linspace(beta_1, beta_T, T)  # betas
    alphas = 1 - betas  # alphas
    alphas_bar = torch.cumprod(alphas, dim=0)  # alpha一把
    sqrt_alphas_bar = torch.sqrt(alphas_bar).to(device)  # 根号下alpha一把
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar).to(device)

    epochs = cfg.epochs
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        tqdm_data_loader = tqdm(data_loader)
        "---------------------"
        for i, (x_0) in enumerate(tqdm_data_loader):  # 由数据加载器加载数据，
            x_0 = x_0.to(device)  # 将数据加载至相应的运行设备(device)
            t = torch.randint(1, T, size=(x_0.shape[0],), device=device)  # 对每一张图片随机在1~T的扩散步中进行采样
            sqrt_alpha_t_bar = torch.gather(sqrt_alphas_bar, dim=0, index=t).reshape(-1, 1, 1, 1)  # 取得不同t下的 根号下alpha_t的连乘
            """取得不同t下的 根号下的一减alpha_t的连乘"""
            sqrt_one_minus_alpha_t_bar = torch.gather(sqrt_one_minus_alphas_bar, dim=0, index=t).reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x_0).to(device)  # 从标准正态分布中采样得到z
            x_t = sqrt_alpha_t_bar * x_0 + sqrt_one_minus_alpha_t_bar * noise  # 计算x_t
            out = model(x_t, t)  # 将x_t输入模型，得到输出
            loss = loss_function(out, noise)  # 将模型的输出，同添加的噪声做损失
            optimizer.zero_grad()  # 优化器的梯度清零
            loss.backward()  # 由损失反向求导
            optimizer.step()  # 优化器更新参数
            "---------------------"
            tqdm_data_loader.set_description(f"Epoch:{epoch}")
            loss_sum += loss.item()
            tqdm_data_loader.set_postfix(ordered_dict={
                "batch": f"{i}/{len(tqdm_data_loader)}",
                "loss": loss_sum/(i+1)*10000,
            })
            time.sleep(0.1)
        if epoch > 0 and epoch % cfg.save_freq == 0:
            torch.save(model.state_dict(), f'{ckpt_dir}/ckpt_{epoch}.pt')  # 保存模型参数


def main():
    args = get_parser()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = './outputs'
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    args.exp_dir = os.path.join(args.exp_dir, f'exp-{args.time_str}')
    cfg = get_config()
    train(args, cfg)


if __name__ == '__main__':
    main()
