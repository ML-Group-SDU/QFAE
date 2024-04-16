import gym
import d4rl
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
import yaml

from utils.buffer import OfflineReplayBuffer
from value.critic import QSarsaLearner
from policy.Behavior_Clone import BehaviorReinforcement

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--configs", default="configs/halfcheetah-medium-v2.yaml", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    args = parser.parse_args()

    with open(args.configs, "r") as data:
        configs = yaml.load(data, Loader=yaml.FullLoader)

    print(f'------current env {configs["meta_data"]["env"]} and current seed {args.seed}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(configs["meta_data"]["path"], configs["meta_data"]["env"], str(args.seed))
    os.makedirs(os.path.join(path, current_time))

    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    env = gym.make(configs["meta_data"]["env"])

    # seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dim of state and action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # offline dataset to replay buffer
    dataset = env.get_dataset()
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']))
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(configs["Q_function"]["gamma"])
    mean, std = replay_buffer.normalize_state()

    # summarywriter logger
    rho = configs["policy"]["rho"]
    comment = configs["meta_data"]["env"] + '_' + str(args.seed)
    log_name = 'bc_' + str(rho)
    logger_path = os.path.join(path, log_name)
    logger = SummaryWriter(log_dir=logger_path, comment=comment)

    # training Q-function of Behavior Cloning
    Q_bc = QSarsaLearner(device, state_dim, action_dim, configs["Q_function"]["hidden_dim"],
                         configs["Q_function"]["depth"], configs["Q_function"]["lr"],
                         configs["Q_function"]["target_update_freq"], configs["Q_function"]["tau"],
                         configs["Q_function"]["gamma"], configs["Q_function"]["q_batch_size"])

    Q_bc_path = os.path.join(path, 'Q_bc.pt')
    if os.path.exists(Q_bc_path):
        Q_bc.load(Q_bc_path)
    else:
        for step in tqdm(range(int(configs["Q_function"]["steps"])), desc='Q_bc updating ......'):
            Q_bc_loss = Q_bc.update(replay_buffer, pi=None)

            if step % int(configs["meta_data"]["log_freq"]) == 0:
                print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
                logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step + 1))

        Q_bc.save(Q_bc_path)

    # log of policy training
    best_bc_pt = 'bc_best_' + str(rho) + '.pt' # best model of behavior cloning
    best_bc_csv = 'best_bc_' + str(rho) + '.csv' # best result of behavior cloning
    last_bc_csv = 'last_bc_' + str(rho) + '.csv' # last result of behavior cloning

    # train Behavior policy with QFAE
    policy = BehaviorReinforcement(device, state_dim, configs["policy"]["hidden_dim"], configs["policy"]["depth"],
                               action_dim, configs["policy"]["lr"], configs["policy"]["batch_size"], rho)
    best_score = 0
    eval_returns = []
    best_bc_path = os.path.join(path, best_bc_pt)

    for step in tqdm(range(int(configs["policy"]["steps"])), desc='policy updating ......'):
        policy_loss = policy.update(replay_buffer, Q_bc)
        logger.add_scalar('policy_loss', policy_loss, global_step=(step + 1))
        if step % int(configs["meta_data"]["log_freq"]) == 0:
            current_score = policy.offline_evaluate(configs["meta_data"]["env"], args.seed, mean, std)
            eval_returns.append((step, current_score))
            np.savetxt(os.path.join(path, 'policy_score.txt'), eval_returns, fmt=['%d', '%.1f'])
            if current_score > best_score:
                best_score = current_score
                # save best policy
                policy.save(best_bc_path)
                np.savetxt(os.path.join(path, best_bc_csv), [best_score], fmt='%f', delimiter=',')
            print(f"Step: {step}, Loss: {policy_loss:.4f}, Score: {current_score:.4f}")
            logger.add_scalar('score', current_score, global_step=(step + 1))

    # save last policy
    np.savetxt(os.path.join(path, last_bc_csv), [current_score], fmt='%f', delimiter=',')
    policy.save(best_bc_path)


    logger.close()
