
from pathlib import Path
import os
from data_utils import noniid_partition_horse_zebra, plot_client_distribution

# 设置路径（你实际的数据目录）
root = Path("../CT2MRI/trainA")
horse_files = sorted([root / f for f in os.listdir(root) if f.endswith((".jpg", ".png"))])

root = Path("../CT2MRI/trainB")
zebra_files = sorted([root / f for f in os.listdir(root) if f.endswith((".jpg", ".png"))])

# 非IID分布划分
num_clients = 2
alpha = 0.2
client_data = noniid_partition_horse_zebra(horse_files, zebra_files, num_clients, alpha)

# 可视化
plot_client_distribution(client_data, alpha)
