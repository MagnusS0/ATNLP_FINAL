from ..train import main
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ..data.dataset import SCANDataset
from collections import defaultdict
from ..utils.utils import greedy_decode, oracle_greedy_search, calculate_accuracy


def calculate_length_accuracies(
    model, dataloader, device, src_vocab, tgt_vocab, use_oracle=False
):
    """Calculate accuracies grouped by source and target lengths."""
    src_length_results = defaultdict(list)
    src_length_seq_results = defaultdict(list)
    tgt_length_results = defaultdict(list)
    tgt_length_seq_results = defaultdict(list)

    eos_idx = tgt_vocab.tok2id["<EOS>"]
    bos_idx = tgt_vocab.tok2id["<BOS>"]
    pad_idx = tgt_vocab.tok2id["<PAD>"]

    # Get special token IDs to exclude from length calculation
    special_tokens = {src_vocab.tok2id[t] for t in ["<PAD>", "<BOS>", "<EOS>"]}

    model.eval()

    all_accuracies = []
    all_seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Calculate lengths excluding special tokens for each sequence in batch
            src_lengths = [
                sum(1 for t in seq if t.item() not in special_tokens) for seq in src
            ]
            tgt_lengths = [
                sum(1 for t in seq if t.item() not in special_tokens) for seq in tgt
            ]

            # Getpredictions
            if use_oracle:
                pred = oracle_greedy_search(
                    model, src, eos_idx, bos_idx, pad_idx, tgt, device
                )
            else:
                pred = greedy_decode(model, src, eos_idx, bos_idx, pad_idx, device)

            # Calculate accuracy
            token_accs, seq_accs = calculate_accuracy(
                pred, tgt, pad_idx, eos_idx, by_example=True
            )

            # Store results by length for each sequence in batch
            for idx, (src_len, tgt_len) in enumerate(zip(src_lengths, tgt_lengths)):
                src_length_results[src_len].append(token_accs[idx].item())
                src_length_seq_results[src_len].append(seq_accs[idx].item())
                tgt_length_results[tgt_len].append(token_accs[idx].item())
                tgt_length_seq_results[tgt_len].append(seq_accs[idx].item())

                all_accuracies.append(token_accs[idx].item())
                all_seq_accuracies.append(seq_accs[idx].item())

    # Calculate mean accuracies for each length
    src_accuracies = {k: np.mean(v) for k, v in src_length_results.items()}
    src_seq_accuracies = {k: np.mean(v) for k, v in src_length_seq_results.items()}
    tgt_accuracies = {k: np.mean(v) for k, v in tgt_length_results.items()}
    tgt_seq_accuracies = {k: np.mean(v) for k, v in tgt_length_seq_results.items()}

    overall_acc = np.mean(all_accuracies)
    overall_seq_acc = np.mean(all_seq_accuracies)

    print(
        f"Overall Token Accuracy: {overall_acc:.4f}, Overall Sequence Accuracy: {overall_seq_acc:.4f}"
    )

    return (
        src_accuracies,
        tgt_accuracies,
        src_seq_accuracies,
        tgt_seq_accuracies,
        overall_acc,
        overall_seq_acc,
    )


def plot_length_accuracies(
    accuracies, title, save_path, is_token=True, color="skyblue"
):
    """Create a bar plot of accuracies by length."""
    plt.figure(figsize=(10, 6))

    # Sort lengths and corresponding accuracies
    lengths = sorted(accuracies.keys())
    if isinstance(next(iter(accuracies.values())), tuple):
        values = [
            100 * (accuracies[l][0] if is_token else accuracies[l][1]) for l in lengths
        ]
    else:
        values = [100 * accuracies[l] for l in lengths]
    length_labels = [str(l) for l in lengths]

    # Create bar plot
    bars = plt.bar(length_labels, values, color=color)
    plt.ylim(0, 100)
    plt.xlabel("Length")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add value labels
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_experiment(n_runs=5):
    """Run training with length split analysis."""
    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    size = "length"
    device = hyperparams["device"]

    # Load test dataset for length analysis
    test_dataset = SCANDataset(test_path)
    test_dataloader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4
    )

    results = {}

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")

        # Train model
        model, _, _, _ = main(
            train_path, test_path, size, hyperparams, oracle=True, random_seed=seed
        )

        # Calculate length-based accuracies for both greedy and oracle
        print("\nCalculating greedy search accuracies...")
        (
            src_acc_greedy,
            tgt_acc_greedy,
            _,
            _,
            overall_acc_greedy,
            overall_seq_acc_greedy,
        ) = calculate_length_accuracies(
            model,
            test_dataloader,
            device,
            test_dataset.src_vocab,
            test_dataset.tgt_vocab,
            use_oracle=False,
        )

        print("Calculating oracle search accuracies...")
        (
            src_acc_oracle,
            tgt_acc_oracle,
            src_seq_acc_oracle,
            tgt_seq_acc_oracle,
            overall_acc_oracle,
            overall_seq_acc_oracle,
        ) = calculate_length_accuracies(
            model,
            test_dataloader,
            device,
            test_dataset.src_vocab,
            test_dataset.tgt_vocab,
            use_oracle=True,
        )

        results[f"run{run+1}"] = {
            "src_acc_greedy": src_acc_greedy,
            "tgt_acc_greedy": tgt_acc_greedy,
            "overall_acc_greedy": overall_acc_greedy,
            "overall_seq_acc_greedy": overall_seq_acc_greedy,
            "src_acc_oracle": src_acc_oracle,
            "tgt_acc_oracle": tgt_acc_oracle,
            "src_seq_acc_oracle": src_seq_acc_oracle,
            "tgt_seq_acc_oracle": tgt_seq_acc_oracle,
            "overall_acc_oracle": overall_acc_oracle,
            "overall_seq_acc_oracle": overall_seq_acc_oracle,
        }

        print(f"Run {run+1} complete.")

    # Calculate mean accuracies across runs
    src_acc_greedy = {
        k: np.mean([results[f"run{i+1}"]["src_acc_greedy"][k] for i in range(n_runs)])
        for k in results["run1"]["src_acc_greedy"].keys()
    }
    tgt_acc_greedy = {
        k: np.mean([results[f"run{i+1}"]["tgt_acc_greedy"][k] for i in range(n_runs)])
        for k in results["run1"]["tgt_acc_greedy"].keys()
    }
    overall_acc_greedy = np.mean(
        [results[f"run{i+1}"]["overall_acc_greedy"] for i in range(n_runs)]
    )
    overall_seq_acc_greedy = np.mean(
        [results[f"run{i+1}"]["overall_seq_acc_greedy"] for i in range(n_runs)]
    )
    overall_std_greedy = np.std(
        [results[f"run{i+1}"]["overall_acc_greedy"] for i in range(n_runs)]
    )
    overall_std_seq_greedy = np.std(
        [results[f"run{i+1}"]["overall_seq_acc_greedy"] for i in range(n_runs)]
    )

    src_acc_oracle = {
        k: np.mean([results[f"run{i+1}"]["src_acc_oracle"][k] for i in range(n_runs)])
        for k in results["run1"]["src_acc_oracle"].keys()
    }
    tgt_acc_oracle = {
        k: np.mean([results[f"run{i+1}"]["tgt_acc_oracle"][k] for i in range(n_runs)])
        for k in results["run1"]["tgt_acc_oracle"].keys()
    }
    src_seq_acc_oracle = {
        k: np.mean(
            [results[f"run{i+1}"]["src_seq_acc_oracle"][k] for i in range(n_runs)]
        )
        for k in results["run1"]["src_seq_acc_oracle"].keys()
    }
    tgt_seq_acc_oracle = {
        k: np.mean(
            [results[f"run{i+1}"]["tgt_seq_acc_oracle"][k] for i in range(n_runs)]
        )
        for k in results["run1"]["tgt_seq_acc_oracle"].keys()
    }
    overall_acc_oracle = np.mean(
        [results[f"run{i+1}"]["overall_acc_oracle"] for i in range(n_runs)]
    )
    overall_seq_acc_oracle = np.mean(
        [results[f"run{i+1}"]["overall_seq_acc_oracle"] for i in range(n_runs)]
    )
    overall_std_oracle = np.std(
        [results[f"run{i+1}"]["overall_acc_oracle"] for i in range(n_runs)]
    )
    overall_std_seq_oracle = np.std(
        [results[f"run{i+1}"]["overall_seq_acc_oracle"] for i in range(n_runs)]
    )

    # Print mean results
    print("\nMean Results:")
    print(
        f"Greedy Search - Mean Token Acc: {overall_acc_greedy:.4f} ± {overall_std_greedy:.4f}, Seq Acc: {overall_seq_acc_greedy:.4f} ± {overall_std_seq_greedy:.4f}"
    )
    print(
        f"Oracle Search - Mean Token Acc: {overall_acc_oracle:.4f} ± {overall_std_oracle:.4f}, Seq Acc: {overall_seq_acc_oracle:.4f} ± {overall_std_seq_oracle:.4f}"
    )
    print(f"Greedy Search - Source Lengths: {src_acc_greedy}")
    print(f"Greedy Search - Target Lengths: {tgt_acc_greedy}")
    print(f"Oracle Search - Source Lengths: {src_acc_oracle}")
    print(f"Oracle Search - Target Lengths: {tgt_acc_oracle}")
    print(f"Oracle Search - Source Lengths (Seq): {src_seq_acc_oracle}")
    print(f"Oracle Search - Target Lengths (Seq): {tgt_seq_acc_oracle}")

    # 1. Action + Greedy + Token
    plot_length_accuracies(
        src_acc_greedy,
        f"Token Accuracy (Greedy) by Action Sequence Length",
        f"experiments/exp2_src_greedy_token_run{run+1}.png",
        is_token=True,
        color="#1f77b4",
    )

    # 2. Command + Greedy + Token
    plot_length_accuracies(
        tgt_acc_greedy,
        f"Token Accuracy (Greedy) by Command Length",
        f"experiments/exp2_tgt_greedy_token_run{run+1}.png",
        is_token=True,
        color="#1f77b4",
    )

    # 3. Source + Oracle + Token
    plot_length_accuracies(
        src_acc_oracle,
        f"Token Accuracy (Oracle) by Action Sequence Length",
        f"experiments/exp2_src_oracle_token_run{run+1}.png",
        is_token=True,
        color="#1f77b4",
    )

    # 4. Target + Oracle + Token
    plot_length_accuracies(
        tgt_acc_oracle,
        f"Token Accuracy (Oracle) by Command Length",
        f"experiments/exp2_tgt_oracle_token_run{run+1}.png",
        is_token=True,
        color="#1f77b4",
    )

    # 5. Source + Oracle + Sequence
    plot_length_accuracies(
        src_seq_acc_oracle,
        f"Sequence Accuracy (Oracle) by Action Sequence Length",
        f"experiments/exp2_src_oracle_seq_run{run+1}.png",
        is_token=False,
        color="#1f77b4",
    )

    # 6. Target + Oracle + Sequence
    plot_length_accuracies(
        tgt_seq_acc_oracle,
        f"Sequence Accuracy (Oracle) by Command Length",
        f"experiments/exp2_tgt_oracle_seq_run{run+1}.png",
        is_token=False,
        color="#1f77b4",
    )

    print(f"All plots for run {run+1} saved to experiments/")


if __name__ == "__main__":
    run_experiment()
