"""Evaluation script for in-repo agents."""

import argparse
import os
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

from algorithms import PPOAgent, SACAgent, TD3Agent
from config import EnvConfig
from racecar_env import RaceCarEnv
from wrappers import (
    BaselineWrapper,
    TD3HeadingWrapper,
    PPOPlannerWrapper,
    SACEndToEndWrapper,
)
from train import recommended_scenario_for_algorithm


def evaluate(
    model_path: Optional[str],
    algorithm: Optional[str] = None,
    num_episodes: int = 10,
    render: bool = True,
    env_config: Optional[EnvConfig] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Modeli değerlendir

    Args:
        model_path: Model dosya yolu
        num_episodes: Test episode sayısı
        render: Görselleştirme
        env_config: Environment konfigürasyonu
        deterministic: Deterministic policy

    Returns:
        Değerlendirme metrikleri
    """
    if env_config is None:
        env_config = EnvConfig()

    baseline_run = _is_baseline_run(algorithm, env_config, model_path)
    if baseline_run and env_config.scenario != "baseline":
        env_config.scenario = "baseline"

    render_mode = "human" if render else None
    base_env = RaceCarEnv(render_mode=render_mode, config=env_config)

    if env_config.scenario == "baseline":
        env = BaselineWrapper(base_env)
    elif env_config.scenario == "td3_heading_control":
        env = TD3HeadingWrapper(base_env)
    elif env_config.scenario == "ppo_trajectory_planner":
        env = PPOPlannerWrapper(base_env)
    else:
        env = SACEndToEndWrapper(base_env)

    if baseline_run:
        if model_path is not None and os.path.exists(model_path):
            print("[Warning] Baseline run ignores the provided model path.")
        print("Baseline controller (pure pursuit + PID).")
        policy = lambda _obs: np.zeros(env.action_space.shape, dtype=np.float32)
    else:
        algo_name = _resolve_algorithm_name(model_path, algorithm)
        model = _build_agent_for_env(algo_name, env)
        if model_path is None:
            raise ValueError("Model path is required for RL evaluation.")
        model.load(model_path)
        print(f"Model loaded: {model_path} ({algo_name})")
        policy = lambda obs: model.act(obs, deterministic=deterministic)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    success_count = 0

    print(f"\nEvaluating {num_episodes} episodes...")
    print("-" * 50)

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if info.get("goal_reached", False):
            success_count += 1
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"

        print(
            f"Episode {ep+1:3d}: {status} | "
            f"Reward: {total_reward:8.2f} | "
            f"Steps: {steps:4d} | "
            f"Dist: {info['distance_to_goal']:.3f}m"
        )

    env.close()

    # Özet
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)

    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / num_episodes * 100,
        "num_episodes": num_episodes,
    }

    print(
        f"Mean Reward:  {metrics['mean_reward']:.2f} (+/- {metrics['std_reward']:.2f})"
    )
    print(
        f"Mean Length:  {metrics['mean_length']:.1f} (+/- {metrics['std_length']:.1f})"
    )
    print(f"Success Rate: {metrics['success_rate']:.1f}%")

    return metrics


def compare_models(
    model_paths: List[str],
    algorithms: Optional[List[str]] = None,
    num_episodes: int = 10,
    env_config: Optional[EnvConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Birden fazla modeli karşılaştır

    Args:
        model_paths: Model dosya yolları listesi
        num_episodes: Her model için test episode sayısı
        env_config: Environment konfigürasyonu

    Returns:
        Model başına metrikler
    """
    results = {}

    for i, path in enumerate(model_paths):
        model_name = os.path.basename(path)
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print("=" * 60)

        metrics = evaluate(
            model_path=path,
            algorithm=algorithms[i] if algorithms is not None else None,
            num_episodes=num_episodes,
            render=False,
            env_config=env_config,
        )

        results[model_name] = metrics

    # Karşılaştırma tablosu
    print(f"\n{'='*60}")
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<30} {'Reward':>12} {'Success':>10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(
            f"{name:<30} {metrics['mean_reward']:>12.2f} {metrics['success_rate']:>9.1f}%"
        )

    return results


def _resolve_algorithm_name(model_path: Optional[str], algorithm: Optional[str]) -> str:
    if algorithm is not None:
        return algorithm.upper()
    if model_path is None:
        raise ValueError("Cannot infer algorithm without a model path.")
    path = model_path.lower()
    if "ppo" in path:
        return "PPO"
    if "sac" in path:
        return "SAC"
    if "td3" in path:
        return "TD3"
    raise ValueError("Cannot infer algorithm from path. Please pass --algorithm.")


def _build_agent_for_env(algorithm: str, env: gym.Env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    if algorithm == "PPO":
        return PPOAgent(observation_dim=obs_dim, action_dim=act_dim)
    if algorithm == "SAC":
        return SACAgent(observation_dim=obs_dim, action_dim=act_dim)
    if algorithm == "TD3":
        return TD3Agent(observation_dim=obs_dim, action_dim=act_dim)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _is_baseline_run(
    algorithm: Optional[str], env_config: EnvConfig, model_path: Optional[str]
) -> bool:
    if algorithm is not None and algorithm.upper() == "BASELINE":
        return True
    if env_config.scenario == "baseline":
        return True
    if model_path is not None and "baseline" in model_path.lower():
        return True
    return False


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Evaluate RaceCar RL Agent")

    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model path (omit for BASELINE)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["BASELINE", "PPO", "SAC", "TD3"],
        help="Algorithm type. If omitted, inferred from model filename.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic policy"
    )
    parser.add_argument("--arena-size", type=float, default=20.0, help="Arena size")
    parser.add_argument(
        "--goal-threshold", type=float, default=0.50, help="Goal threshold"
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

    args = parser.parse_args()

    if args.algorithm is None and args.model is None and args.scenario != "baseline":
        parser.error(
            "Please provide --algorithm, a model path, or --scenario baseline."
        )

    algorithm = args.algorithm
    if algorithm is None and args.model is not None:
        algorithm = _resolve_algorithm_name(args.model, None)
    if algorithm is None and args.scenario == "baseline":
        algorithm = "BASELINE"

    scenario = args.scenario or recommended_scenario_for_algorithm(algorithm)
    if algorithm == "BASELINE" and scenario != "baseline":
        print("[Warning] BASELINE selected; forcing scenario to 'baseline'.")
        scenario = "baseline"
    if scenario == "baseline" and algorithm != "BASELINE":
        print("[Warning] Baseline scenario ignores the selected algorithm.")
    if (
        args.scenario is not None
        and args.scenario != recommended_scenario_for_algorithm(algorithm)
    ):
        print(
            f"[Warning] Scenario '{args.scenario}' differs from recommended "
            f"'{recommended_scenario_for_algorithm(algorithm)}' for {algorithm}; using user choice."
        )

    env_config = EnvConfig(
        arena_size=args.arena_size,
        goal_threshold=args.goal_threshold,
        scenario=scenario,
    )

    evaluate(
        model_path=args.model,
        algorithm=algorithm,
        num_episodes=args.episodes,
        render=not args.no_render,
        env_config=env_config,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
