"""Training script using in-repo reference algorithms."""

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from algorithms import PPOAgent, SACAgent, TD3Agent
from algorithms.buffers import PPOReplayBuffer, ReplayBuffer, RolloutBuffer
from config import CurriculumConfig, EnvConfig, TrainingConfig
from racecar_env import RaceCarEnv
from wrappers import (
    CurriculumWrapper,
    BaselineWrapper,
    TD3HeadingWrapper,
    PPOPlannerWrapper,
    SACEndToEndWrapper,
)


def create_env(config: EnvConfig, render_mode: Optional[str] = None) -> gym.Env:
    import gymnasium as gym

    base_env = RaceCarEnv(render_mode=render_mode, config=config)
    if config.scenario == "baseline":
        return BaselineWrapper(base_env)
    elif config.scenario == "td3_heading_control":
        return TD3HeadingWrapper(base_env)
    elif config.scenario == "ppo_trajectory_planner":
        return PPOPlannerWrapper(base_env)
    else:
        return SACEndToEndWrapper(base_env)


def recommended_scenario_for_algorithm(algorithm: str) -> str:
    mapping = {
        "BASELINE": "baseline",
        "PPO": "ppo_trajectory_planner",
        "SAC": "sac_end_to_end",
        "TD3": "td3_heading_control",
    }
    return mapping[algorithm.upper()]


def train(
    config: Optional[TrainingConfig] = None,
    env_config: Optional[EnvConfig] = None,
    use_curriculum: bool = False,
    curriculum_config: Optional[CurriculumConfig] = None,
):
    if config is None:
        config = TrainingConfig()
    if env_config is None:
        env_config = EnvConfig()
    algo = config.algorithm.upper()
    if algo == "BASELINE" and env_config.scenario != "baseline":
        print(
            "[Warning] BASELINE selected; forcing scenario to 'baseline' (no training)."
        )
        env_config.scenario = "baseline"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    print("=" * 60)
    print("RaceCar RL Training")
    print("=" * 60)
    print(f"Algorithm: {config.algorithm}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Run directory: {run_dir}")

    env = create_env(env_config, render_mode=None)
    if use_curriculum:
        env = CurriculumWrapper(env, curriculum_config)
    eval_env = create_env(env_config, render_mode=None)

    if env_config.scenario == "baseline":
        if algo != "BASELINE":
            print(
                f"[Warning] Scenario 'baseline' ignores algorithm '{config.algorithm}'."
            )
        _run_baseline(env, episodes=config.baseline_episodes)
        env.close()
        eval_env.close()
        return None

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = _build_agent(config, obs_dim, act_dim)

    print("\nStarting training...")
    if algo == "PPO":
        _train_ppo(agent, env, eval_env, config, run_dir)
    else:
        _train_offpolicy(agent, env, eval_env, config, run_dir)

    final_model_path = os.path.join(
        config.model_dir, f"racecar_{algo.lower()}_final.pt"
    )
    agent.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    env.close()
    eval_env.close()
    return agent


def _build_agent(config: TrainingConfig, obs_dim: int, act_dim: int):
    algo = config.algorithm.upper()
    if algo == "PPO":
        return PPOAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            lr=config.learning_rate,
            clip_range=config.clip_range,
            entropy_coef=config.ent_coef,
        )
    if algo == "SAC":
        return SACAgent(
            observation_dim=obs_dim, action_dim=act_dim, actor_lr=config.learning_rate
        )
    if algo == "TD3":
        return TD3Agent(
            observation_dim=obs_dim, action_dim=act_dim, actor_lr=config.learning_rate
        )
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def _run_eval(agent, env: gym.Env, episodes: int = 3) -> Tuple[float, float]:
    rewards = []
    successes = 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        successes += int(info.get("goal_reached", False))
    return float(np.mean(rewards)), 100.0 * successes / episodes


def _run_baseline(env: gym.Env, episodes: int = 5) -> None:
    print("\nBaseline run (pure pursuit + PID). No training performed.")
    rewards = []
    successes = 0
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        successes += int(info.get("goal_reached", False))
        print(
            f"Episode {ep + 1:3d} | reward={total_reward:8.2f} | steps={steps:4d} | goal={info.get('goal_reached', False)}"
        )

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    success_rate = 100.0 * successes / max(1, episodes)
    print("-" * 50)
    print(f"Baseline mean reward: {mean_reward:.2f}")
    print(f"Baseline success rate: {success_rate:.1f}%")


def _train_offpolicy(
    agent, env, eval_env, config: TrainingConfig, run_dir: str
) -> None:
    obs, _ = env.reset()
    replay = ReplayBuffer.create(
        capacity=max(100000, config.total_timesteps),
        obs_dim=obs.shape[0],
        act_dim=env.action_space.shape[0],
    )
    episode_reward = 0.0
    episode_count = 0
    warmup_steps = max(1000, config.batch_size * 5)

    for step in range(1, config.total_timesteps + 1):
        if step < warmup_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_done = terminated or truncated
        replay.add(obs, action, reward, next_obs, episode_done)
        obs = next_obs
        episode_reward += reward

        if replay.size >= config.batch_size:
            metrics = agent.train_step(replay.sample(config.batch_size))
        else:
            metrics = {}

        if episode_done:
            episode_count += 1
            print(
                f"Episode {episode_count:4d} | reward={episode_reward:8.2f} | goal={info.get('goal_reached', False)}"
            )
            obs, _ = env.reset()
            episode_reward = 0.0

        if step % config.eval_freq == 0:
            mean_reward, success = _run_eval(agent, eval_env)
            print(
                f"[Eval step {step}] mean_reward={mean_reward:.2f}, success={success:.1f}%"
            )

        if step % config.save_freq == 0:
            ckpt_path = os.path.join(
                run_dir, f"racecar_{config.algorithm.lower()}_{step}.pt"
            )
            agent.save(ckpt_path)
            if metrics:
                print(f"[Checkpoint] {ckpt_path} | {metrics}")


def _train_ppo(
    agent: PPOAgent, env, eval_env, config: TrainingConfig, run_dir: str
) -> None:
    obs, _ = env.reset()
    rollout = RolloutBuffer(
        capacity=config.n_steps,
        obs_dim=int(np.prod(env.observation_space.shape)),
        act_dim=int(np.prod(env.action_space.shape)),
    )
    replay = PPOReplayBuffer.create(
        capacity=config.ppo_replay_capacity,
        obs_dim=int(np.prod(env.observation_space.shape)),
        act_dim=int(np.prod(env.action_space.shape)),
    )
    total_steps = 0
    while total_steps < config.total_timesteps:
        rollout.reset()

        for _ in range(config.n_steps):
            action, log_prob, value = agent.sample_action(obs)
            for _ in range(3):
                next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated
            next_value = 0.0 if terminated else agent.value(next_obs)
            rollout.add(
                obs=obs,
                action=action,
                reward=float(reward),
                terminated=terminated,
                episode_end=episode_done,
                value=float(value),
                log_prob=float(log_prob),
                next_value=float(next_value),
            )
            obs = next_obs
            total_steps += 1
            if episode_done:
                obs, _ = env.reset()
            if total_steps >= config.total_timesteps:
                break

        advantages, returns = rollout.compute_advantages(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        batch = rollout.as_batch(advantages, returns)
        replay.add_batch(batch)

        if replay.size < config.batch_size:
            continue

        metrics = {}
        for _ in range(config.n_epochs):
            num_batches = max(1, replay.size // config.batch_size)
            for _ in range(num_batches):
                mini_batch = replay.sample(config.batch_size)
                metrics = agent.train_step(mini_batch)
        if total_steps % config.eval_freq < config.n_steps:
            mean_reward, success = _run_eval(agent, eval_env)
            # print(
            #     f"[Eval step {total_steps}] mean_reward={mean_reward:.2f}, success={success:.1f}% | {metrics}"
            # )
            print(
                f"[Eval step {total_steps}] mean_reward={mean_reward:.2f}, success={success:.1f}%"
            )
        if total_steps % config.save_freq < config.n_steps:
            ckpt_path = os.path.join(run_dir, f"racecar_ppo_{total_steps}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] {ckpt_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Train RaceCar RL Agent")

    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["BASELINE", "PPO", "SAC", "TD3"],
        help="RL algorithm",
    )
    parser.add_argument(
        "--timesteps", type=int, default=400000, help="Total training timesteps"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--curriculum", action="store_true", help="Use curriculum learning"
    )
    parser.add_argument(
        "--arena-size", type=float, default=20.0, help="Arena size in meters"
    )
    parser.add_argument(
        "--goal-threshold", type=float, default=0.50, help="Goal threshold in meters"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=[
            "baseline",
            "sac_end_to_end",
            "td3_heading_control",
            "ppo_trajectory_planner",
        ],
        help="Homework scenario/control stack",
    )
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=TrainingConfig().baseline_episodes,
        help="Episodes to run in baseline mode",
    )
    parser.add_argument(
        "--ppo-replay-capacity",
        type=int,
        default=TrainingConfig().ppo_replay_capacity,
        help="Replay buffer capacity for PPO",
    )

    args = parser.parse_args()

    # Create training configuration
    train_config = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        baseline_episodes=args.baseline_episodes,
        ppo_replay_capacity=args.ppo_replay_capacity,
    )

    scenario = args.scenario or recommended_scenario_for_algorithm(args.algorithm)
    if (
        args.scenario is not None
        and args.scenario != recommended_scenario_for_algorithm(args.algorithm)
    ):
        print(
            f"[Warning] Scenario '{args.scenario}' differs from recommended "
            f"'{recommended_scenario_for_algorithm(args.algorithm)}' for {args.algorithm}; using user choice."
        )

    env_config = EnvConfig(
        arena_size=args.arena_size,
        goal_threshold=args.goal_threshold,
        scenario=scenario,
    )

    curriculum_config = CurriculumConfig() if args.curriculum else None

    # Eğitimi başlat
    train(
        config=train_config,
        env_config=env_config,
        use_curriculum=args.curriculum,
        curriculum_config=curriculum_config,
    )


if __name__ == "__main__":
    main()
