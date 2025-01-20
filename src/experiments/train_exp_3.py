import numpy as np
from ..train import main
import torch
from rich import print
from rich.traceback import install
import matplotlib.pyplot as plt

install()


def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "data/add_prim_split"

    # Add the num0 dataset explicitly
    pairs = [
        # (
        #     f"{base_path}/tasks_train_addprim_jump.txt",
        #     f"{base_path}/tasks_test_addprim_jump.txt",
        #     "jump",
        # ),
        (
            f"{base_path}/tasks_train_addprim_turn_left.txt",
            f"{base_path}/tasks_test_addprim_turn_left.txt",
            "turn_left",
        ),
    ]

    additional_base_path = "data/add_prim_split/with_additional_examples"
    num_composed_commands = ["num1", "num2", "num4", "num8", "num16", "num32"]
    for num in num_composed_commands:
        train_test_pairs = []
        for rep in range(1, 6):  # Changed to 5 repetitions
            train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}_rep{rep}.txt"
            test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}_rep{rep}.txt"
            train_test_pairs.append((train_path, test_path))
        pairs.append((train_test_pairs, num))

    return pairs


def plot_jump_accuracies(jump_results, save_path_prefix):
    """Create bar plots for jump accuracies."""
    # Prepare data
    nums = ["0"] + [result[0].replace("num", "") for result in jump_results[1:]]
    token_accs = []
    seq_accs = []
    token_stds = []
    seq_stds = []

    # Process jump (num0) result
    jump_accs = np.array(
        [
            (
                acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
            )
            for acc, g_acc in jump_results[0]
        ]
    )
    token_accs.append(np.mean(jump_accs[:, 0]))
    seq_accs.append(np.mean(jump_accs[:, 1]))
    token_stds.append(np.std(jump_accs[:, 0]))
    seq_stds.append(np.std(jump_accs[:, 1]))

    # Process numerical results
    for result in jump_results[1:]:
        accs = np.array(
            [
                (
                    acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                    g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
                )
                for acc, g_acc in result[1]
            ]
        )
        token_accs.append(np.mean(accs[:, 0]))
        seq_accs.append(np.mean(accs[:, 1]))
        token_stds.append(np.std(accs[:, 0]))
        seq_stds.append(np.std(accs[:, 1]))

    # Create token accuracy plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(nums))
    plt.bar(x, token_accs, yerr=token_stds, capsize=5)
    plt.xlabel("Number of Composed Commands Used for Training")
    plt.ylabel("Accuracy on new commands (%)")
    plt.title("Token Accuracy")
    plt.xticks(x, nums)
    plt.grid(True, axis="y")

    # Add value labels
    for i, (v, std) in enumerate(zip(token_accs, token_stds)):
        plt.text(i, v + std, f"{v:.3f}±{std:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_token.png")
    plt.close()

    # Create sequence accuracy plot
    plt.figure(figsize=(12, 6))
    plt.bar(x, seq_accs, yerr=seq_stds, capsize=5)
    plt.xlabel("Number of Composed Commands Used for Training")
    plt.ylabel("Accuracy on new commands (%)")
    plt.title("Sequence Accuracy")
    plt.xticks(x, nums)
    plt.grid(True, axis="y")

    # Add value labels
    for i, (v, std) in enumerate(zip(seq_accs, seq_stds)):
        plt.text(i, v + std, f"{v:.3f}±{std:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_seq.png")
    plt.close()

    return token_accs, seq_accs, token_stds, seq_stds


def run_experiment_3(n_runs=5):
    """
    Run Experiment 3: Adding a new primitive and testing generalization to composed commands.
    Uses n_runs for basic cases (jump, turn_left) and existing 5 repetitions for numerical cases.
    """
    jump_results = []
    turn_left_results = None

    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Fetch dataset pairs
    pairs = get_add_prim_dataset_pairs()

    # Process jump (num0)
    train_path, test_path, name = pairs[0]
    print(f"\nProcessing jump (num0)")
    print("=" * 70)

    basic_results = []
    for run in range(n_runs):
        seed = 42 + run
        print(f"Run {run+1}/{n_runs} with seed {seed}")
        _, accuracy, g_accuracy, plot_data = main(
            train_path, test_path, name, hyperparams, random_seed=seed, oracle=False
        )
        basic_results.append(
            (plot_data["greedy_accuracies"][0], plot_data["greedy_seq_accuracies"][0])
        )
    jump_results.append(basic_results)

    # Process turn_left
    train_path, test_path, name = pairs[1]
    print(f"\nProcessing turn_left")
    print("=" * 70)

    turn_left_results = []
    for run in range(n_runs):
        seed = 42 + run
        print(f"Run {run+1}/{n_runs} with seed {seed}")
        _, accuracy, g_accuracy, plot_data = main(
            train_path, test_path, name, hyperparams, random_seed=seed, oracle=False
        )
        turn_left_results.append(
            (plot_data["greedy_accuracies"][0], plot_data["greedy_seq_accuracies"][0])
        )

    # Process the numerical cases (using existing 5 repetitions)
    for train_test_pairs, num in pairs[2:]:
        print(f"\nProcessing dataset {num}")
        print("=" * 70)

        rep_results = []
        for train_path, test_path in train_test_pairs:
            _, accuracy, g_accuracy, plot_data = main(
                train_path, test_path, num, hyperparams, random_seed=42, oracle=False
            )
            rep_results.append(
                (
                    plot_data["greedy_accuracies"][0],
                    plot_data["greedy_seq_accuracies"][0],
                )
            )
        jump_results.append((num, rep_results))

    # Create plots for jump results
    token_accs, seq_accs, token_stds, seq_stds = plot_jump_accuracies(
        jump_results, "experiments/exp3_jump"
    )

    # Calculate turn_left averages
    turn_left_accs = np.array(
        [
            (
                acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
            )
            for acc, g_acc in turn_left_results
        ]
    )
    turn_left_token_mean = np.mean(turn_left_accs[:, 0])
    turn_left_seq_mean = np.mean(turn_left_accs[:, 1])
    turn_left_token_std = np.std(turn_left_accs[:, 0])
    turn_left_seq_std = np.std(turn_left_accs[:, 1])

    # Print summary
    print("\nFinal Results Summary:")
    print("=" * 50)

    print("\nJump Results:")
    for i, num in enumerate(["num0"] + [r[0] for r in jump_results[1:]]):
        print(f"{num:8} | Token Acc: {token_accs[i]:.4f} ± {token_stds[i]:.4f}")
        print(f"        | Seq Acc:   {seq_accs[i]:.4f} ± {seq_stds[i]:.4f}")

    print("\nTurn Left Results:")
    print(f"Token Accuracy: {turn_left_token_mean:.4f} ± {turn_left_token_std:.4f}")
    print(f"Sequence Accuracy: {turn_left_seq_mean:.4f} ± {turn_left_seq_std:.4f}")

    print(
        f"\nPlots saved to experiments/exp3_jump_token.png and experiments/exp3_jump_seq.png"
    )


if __name__ == "__main__":
    run_experiment_3()
