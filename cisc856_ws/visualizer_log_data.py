#!/usr/bin/env python3

import re
import csv
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


LOG_PATTERN = re.compile(
    r"Total Epi:\s*(\d+)\s+"
    r"Steps:\s*(\d+)\s+"
    r"Episode Steps:\s*(\d+)\s+"
    r"Return:\s*([-+]?\d+\.\d+)\s+"
    r"Coverage Node:\s*([-+]?\d+\.\d+)\s+"
    r"Coverage Edge\s*([-+]?\d+\.\d+)\s+"
    r"Eps:\s*([-+]?\d+\.\d+)"
)


def parse_log(logfile):
    data = []

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)

            if match:
                episode_steps = int(match.group(3))
                return_value = float(match.group(4))

                # Nachträgliche Korrektur:
                if episode_steps == 1250:
                    return_value -= 50

                data.append({
                    "episode": int(match.group(1)),
                    "steps": int(match.group(2)),
                    "episode_steps": int(match.group(3)),
                    "return": float(match.group(4)),
                    "coverage_node": float(match.group(5)),
                    "coverage_edge": float(match.group(6)),
                    "coverage_overall": (float(match.group(5)) + float(match.group(6))) / 2,
                    "coverage_per_step": ((float(match.group(5)) + float(match.group(6))) / 2)  * 144 / int(match.group(3)),
                    "epsilon": float(match.group(7))
                })

    return data


def save_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "steps",
                "episode_steps",
                "return",
                "coverage_node",
                "coverage_edge",
                "coverage_overall",
                "coverage_per_step",
                "epsilon"
            ]
        )

        writer.writeheader()
        writer.writerows(data)


def moving_average(values, window=20):
    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        subset = values[start:i + 1]
        result.append(sum(subset) / len(subset))

    return result


def create_plots(data):
    if not HAS_MATPLOTLIB:
        print("matplotlib nicht installiert. Überspringe Plots.")
        return

    episodes = [d["episode"] for d in data]
    returns = [d["return"] for d in data]
    coverage_node = [d["coverage_node"] for d in data]
    coverage_edge = [d["coverage_edge"] for d in data]
    episode_steps = [d["episode_steps"] for d in data]

    returns_ma = moving_average(returns, 20)
    coverage_node_ma = moving_average(coverage_node, 20)
    coverage_edge_ma = moving_average(coverage_edge, 20)
    episode_steps_ma = moving_average(episode_steps, 20)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Return
    axes[0].plot(episodes, returns, label="Return")
    axes[0].plot(episodes, returns_ma, linewidth=2, label="MA20")
    axes[0].set_title("Training Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)
    axes[0].legend()

    # Coverage Node
    axes[1].plot(episodes, coverage_node, label="Node Coverage")
    axes[1].plot(episodes, coverage_node_ma, linewidth=2, label="MA20")
    axes[1].set_title("Coverage Node")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage")
    axes[1].grid(True)

    # Coverage Edge
    axes[2].plot(episodes, coverage_edge, label="Edge Coverage")
    axes[2].plot(episodes, coverage_edge_ma, linewidth=2, label="MA20")
    axes[2].set_title("Coverage Edge")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coverage")
    axes[2].grid(True)

    # Steps per Episode
    axes[3].plot(episodes, episode_steps, label="Steps")
    axes[3].plot(episodes, episode_steps_ma, linewidth=2, label="MA20")
    axes[3].set_title("Steps per Episode")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Steps")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300)
    plt.show()


def print_summary(data):
    if not data:
        print("Keine Trainingsdaten gefunden.")
        return

    returns = [d["return"] for d in data]

    print("\n=== Zusammenfassung ===")
    print(f"Episoden:        {len(data)}")
    print(f"Max Return:      {max(returns):.4f}")
    print(f"Min Return:      {min(returns):.4f}")
    print(f"Durchschn. Return: {sum(returns)/len(returns):.4f}")

    print("\nLetzte Episode:")
    print(data[-1])


def main():
    if len(sys.argv) < 2:
        print("Verwendung:")
        print(f"  python {sys.argv[0]} training.log")
        sys.exit(1)

    logfile = sys.argv[1]

    print(f"Lese Logdatei: {logfile}")

    data = parse_log(logfile)

    print(f"{len(data)} Episoden gefunden.")

    save_csv(data, "training_metrics.csv")
    print("CSV gespeichert: training_metrics.csv")

    print_summary(data)

    create_plots(data)


if __name__ == "__main__":
    main()