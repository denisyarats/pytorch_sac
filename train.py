import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch

import dmc, utils
from logger import Logger
from replay_buffer3 import ReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    # create envs
    train_env = dmc.make(cfg.task, cfg.seed)
    eval_env = dmc.make(cfg.task, cfg.seed)

    # create replay buffer
    replay_buffer = ReplayBuffer(specs=train_env.specs(),
                                 max_size=cfg.replay_buffer_size,
                                 batch_size=cfg.batch_size,
                                 nstep=cfg.nstep,
                                 discount=cfg.discount)

    #self.replay_loader = make_replay_loader(self.work_dir / 'buffer',
    #                                        cfg.replay_buffer_size,
    #                                        cfg.batch_size,
    #                                        cfg.replay_buffer_num_workers,
    #                                        cfg.save_snapshot, cfg.nstep,
    #                                        cfg.discount)
    replay_iter = None

    video = VideoRecorder(work_dir if cfg.save_video else None)

    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_dim=train_env.observation_spec().shape[0],
        action_dim=train_env.action_spec().shape[0])

    timer = utils.Timer()

    def eval(step, episode):
        eval_return = 0
        for i in range(cfg.num_eval_episodes):
            time_step = eval_env.reset()
            video.init(eval_env, enabled=True)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(time_step.observation,
                                       step,
                                       eval_mode=True)
                time_step = eval_env.step(action)
                video.record(eval_env)
                eval_return += time_step.reward

            video.save(f'{step}_{i}.mp4')

        with logger.log_and_dump_ctx(step, ty='eval') as log:
            log('episode_return', eval_return / cfg.num_eval_episodes)
            log('episode', episode)
            log('total_time', timer.total_time())

    episode, episode_step, episode_return = 0, 0, 0
    time_step = train_env.reset()
    metrics = None
    for step in range(cfg.num_train_steps + 1):
        if time_step.last():
            episode += 1
            if metrics is not None:
                elapsed_time, total_time = timer.reset()
                with logger.log_and_dump_ctx(step, ty='train') as log:
                    log('fps', episode_step / elapsed_time)
                    log('total_time', total_time)
                    log('episode_return', episode_return)
                    log('episode', episode)
                    log('buffer_size', len(replay_buffer))

            time_step = train_env.reset()
            episode_step, episode_return = 0, 0

        if step % cfg.eval_every_steps == 0:
            eval(step, episode)

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation, step, eval_mode=False)

        if step >= cfg.num_seed_steps:
            if replay_iter is None:
                replay_iter = iter(replay_buffer)
            metrics = agent.update(replay_iter, step)
            logger.log_metrics(metrics, step, ty='train')

        time_step = train_env.step(action)
        episode_return += time_step.reward
        replay_buffer.add(time_step)
        episode_step += 1


if __name__ == '__main__':
    main()
