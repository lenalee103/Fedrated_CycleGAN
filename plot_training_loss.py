import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def compute_avg_per_round(df):
    return df.groupby("round")[["loss_G", "loss_D", "loss_cycle", "loss_identity"]].mean().reset_index()


# 自动读取所有 client_X_loss.csv 文件
csv_files = glob.glob("results_CT2MRI_alpha0.5_FedProx/loss_logs/client_*_loss.csv")

plt.figure(figsize=(12, 6))
colors = {
    "loss_G": "tab:blue",
    "loss_D": "tab:orange",
    "loss_cycle": "tab:green",
    "loss_identity": "tab:red"
}
linestyles = ["-", "--", "-.", ":"]

for idx, filepath in enumerate(csv_files):
    client_id = os.path.basename(filepath).split("_")[1]
    df = pd.read_csv(filepath)
    avg_df = compute_avg_per_round(df)

    for i, loss_name in enumerate(colors):
        plt.plot(
            avg_df["round"],
            avg_df[loss_name],
            label=f"Client {client_id} - {loss_name}",
            color=colors[loss_name],
            linestyle=linestyles[idx % len(linestyles)]
        )

plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("CT2MRI_Average Loss per Round (2 Clients)-alpha:0.5-FedProx")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
