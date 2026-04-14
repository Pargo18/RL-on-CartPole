import argparse
from src.config import HyperParams
from src.training import train


def parse_args():
    parser = argparse.ArgumentParser(description="DQN CartPole Training")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for experience replay")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--target-update", type=int, default=10, help="Target network update frequency")
    return parser.parse_args()


def main():
    args = parse_args()
    params = HyperParams(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        target_update=args.target_update,
    )
    print(f"Starting DQN training on CartPole with {params.num_episodes} episodes")
    episode_durations = train(params)
    print(f"Training complete. Final episode duration: {episode_durations[-1]}")


if __name__ == "__main__":
    main()
