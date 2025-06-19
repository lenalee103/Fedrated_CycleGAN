import os
import json
import matplotlib.pyplot as plt
import csv
# 指定四个文件夹路径（修改为你的实际路径）
folders = [
    "results_CT2MRI_alpha100FedAvg",
    "results_CT2MRI_alpha0.5FedAvg",
    "results_CT2MRI_alpha0.5FedAdam",
    "results_CT2MRI_alpha0.5FedProx"
    # "results_alpha100"
]

plt.figure(figsize=(10, 6))

for folder in folders:

    folder_name = os.path.basename(folder)
    label = folder_name.split("_", 1)[-1] if "_" in folder_name else folder_name

    json_path = os.path.join(folder, "round_loss_log.json")
    csv_path = os.path.join(folder, "Eval_loss.csv")

    losses = []

    if os.path.exists(json_path):
        # JSON 文件处理
        with open(json_path) as f:
            logs = json.load(f)
        losses = [entry.get("eval_loss", 0.0) for entry in logs]

    elif os.path.exists(csv_path):
        # CSV 文件处理（提取第二列）
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # 跳过表头
            for row in reader:
                try:
                    losses.append(float(row[1]))
                except (IndexError, ValueError):
                    continue

    else:
        print(f"⚠️  No round_loss_log.json or .csv found in: {folder}")
        continue

    rounds = list(range(1, len(losses) + 1))
    plt.plot(rounds, losses, label=label)

plt.title("CT2MRI Evaluation Loss per Round (Comparison)")
plt.xlabel("Federated Round")
plt.ylabel("Evaluation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("results/global_test_loss_comparison.png", dpi=300)
plt.show()
