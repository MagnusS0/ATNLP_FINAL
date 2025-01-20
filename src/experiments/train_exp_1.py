from ..train import main
from ..data.dataset import SCANDataset
import torch
import numpy as np
from rich import print
from rich.traceback import install
import matplotlib.pyplot as plt

install()


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def plot_experiment_results(
    sizes,
    mean_tf_accuracies,
    std_tf_accuracies,
    mean_greedy_accuracies,
    std_greedy_accuracies,
    save_path="experiments/exp1_results.png",
):
    """
    Create bar plots for experiment one showing accuracies for different dataset sizes.

    Args:
        sizes: List of dataset sizes
        mean_tf_accuracies: Mean teacher forcing accuracies
        std_tf_accuracies: Standard deviation of teacher forcing accuracies
        mean_greedy_accuracies: Mean greedy search accuracies
        std_greedy_accuracies: Standard deviation of greedy search accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    x = np.arange(len(sizes))
    width = 0.35

    # Create bars with error bars
    plt.bar(
        x - width / 2,
        mean_tf_accuracies,
        width,
        yerr=std_tf_accuracies,
        label="Teacher Forcing",
        capsize=5,
    )
    plt.bar(
        x + width / 2,
        mean_greedy_accuracies,
        width,
        yerr=std_greedy_accuracies,
        label="Greedy Search",
        capsize=5,
    )

    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: Teacher Forcing vs Greedy Search")
    plt.xticks(x, [f"p{s}" for s in sizes])
    plt.legend()

    # Add value labels on top of each bar
    # for i, (v, std) in enumerate(zip(mean_tf_accuracies, std_tf_accuracies)):
    #     plt.text(i - width/2, v + std, f'{v:.3f}±{std:.3f}',
    #             ha='center', va='bottom')
    # for i, (v, std) in enumerate(zip(mean_greedy_accuracies, std_greedy_accuracies)):
    #     plt.text(i + width/2, v + std, f'{v:.3f}±{std:.3f}',
    #             ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_all_variations(n_runs=5):
    """Run training 5 times for all dataset size variations with different seeds"""
    n_runs = n_runs
    results = {}

    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 1,
        "n_heads": 8,
        "forward_dim": 512,
        "dropout": 0.05,
        "learning_rate": 7e-4,
        "batch_size": 64,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    for train_path, test_path, size in get_dataset_pairs():
        results[f"p{size}"] = []

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")
        print("=" * 70)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            _, accuracy, g_accuracy, _ = main(
                train_path,
                test_path,
                f"p_{size}",
                hyperparams,
                random_seed=seed,
            )
            results[f"p{size}"].append((accuracy, g_accuracy))

    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)

    # Prepare data for plotting
    sizes = []
    mean_tf_accuracies = []
    std_tf_accuracies = []
    mean_greedy_accuracies = []
    std_greedy_accuracies = []

    for size, accuracies in results.items():
        sizes.append(size.replace("p", ""))
        accuracies = [
            (
                acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
            )
            for acc, g_acc in accuracies
        ]
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)

        # Store for plotting
        mean_tf_accuracies.append(mean[0])
        std_tf_accuracies.append(std[0])
        mean_greedy_accuracies.append(mean[1])
        std_greedy_accuracies.append(std[1])

        print(f"{size:11} | Mean Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")

    # Create and save the plot
    plot_experiment_results(
        sizes,
        mean_tf_accuracies,
        std_tf_accuracies,
        mean_greedy_accuracies,
        std_greedy_accuracies,
    )
    print("\nExperiment plots saved to experiments/exp1_results.png")


if __name__ == "__main__":
    run_all_variations()
