import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import os


def partition_single_domain_indices(file_list, num_clients, alpha):
    num_samples = len(file_list)
    client_to_indices = defaultdict(list)

    proportions = np.random.dirichlet(alpha=[alpha]*num_clients)
    proportions = (proportions * num_samples).astype(int)

    # 调整总和确保不丢样本
    while proportions.sum() < num_samples:
        proportions[np.random.choice(num_clients)] += 1
    while proportions.sum() > num_samples:
        proportions[np.random.choice(num_clients)] -= 1

    idx = 0
    for client_id, count in enumerate(proportions):
        client_to_indices[client_id] = file_list[idx:idx+count]
        idx += count

    return client_to_indices


def noniid_partition_horse_zebra(horse_files, zebra_files, num_clients, alpha):
    horse_partition = partition_single_domain_indices(horse_files, num_clients, alpha)
    zebra_partition = partition_single_domain_indices(zebra_files, num_clients, alpha)

    client_to_data = {}
    for cid in range(num_clients):
        client_to_data[cid] = {
            "horse": horse_partition.get(cid, []),
            "zebra": zebra_partition.get(cid, [])
        }
    return client_to_data


def plot_client_distribution(client_to_data, alpha=None):
    horse_counts = []
    zebra_counts = []

    for cid in sorted(client_to_data):
        horse_counts.append(len(client_to_data[cid]['horse']))
        zebra_counts.append(len(client_to_data[cid]['zebra']))

    x = np.arange(len(client_to_data))  # client indices
    bar_width = 0.4  # width of each bar

    # 绘制柱状图
    bars1 = plt.bar(x - bar_width/2, horse_counts, width=bar_width, label="CT")
    bars2 = plt.bar(x + bar_width/2, zebra_counts, width=bar_width, label="MRI")

    # 添加数值标签
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval), ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval), ha='center', va='bottom', fontsize=8)

    # 图像美化
    plt.xlabel("Client ID")
    plt.ylabel("# of Samples")
    plt.title(f"Client Data Distribution (alpha={alpha})" if alpha else "Client Data Distribution")
    plt.xticks(x, [str(cid) for cid in sorted(client_to_data)])
    plt.legend()
    plt.tight_layout()
    plt.show()
