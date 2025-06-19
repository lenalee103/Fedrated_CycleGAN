"""fedvaeexample: A Flower / PyTorch app for Federated CycleGAN."""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from fedvaeexample.task import (
    CycleGAN, get_weights, load_data, set_weights, test, train,
    plot_combined_training_curves, PLOTS_DIR
)

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import csv
from pathlib import Path

logger = logging.getLogger(__name__)


class CycleGANClient(NumPyClient):
    def __init__(self, trainloader, testloaders, local_epochs, learning_rate, client_id):
        self.net = CycleGAN()
        self.trainloader = trainloader  # 现在是 paired_train_loader，即 zip(...)
        self.testloaders = testloaders
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        logger.info(f"Initialized Client {client_id}] Using device: {self.device}")

    def fit(self, parameters, config):
        logger.info(f"Client {self.client_id} - Starting training")
        set_weights(self.net, parameters)

        train_loader = self.trainloader  # ✅ 注意：不再是 horse_loader 和 zebra_loader，而是 paired_loader

        history = train(
            net=self.net,
            paired_loader=train_loader,  # ✅ 直接传入打包后的 DataLoader
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            client_id=self.client_id
        )

        logger.info(f"Client {self.client_id} - Training completed")

        # === 保存训练记录 ===
        history_path = PLOTS_DIR / f"client_{self.client_id}_history.pt"
        torch.save(history, history_path)

        # === 写入 CSV ===
        round_id = config.get("round", 0)
        csv_path = Path("./loss_logs") / f"client_{self.client_id}_loss.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["round", "epoch", "loss_G", "loss_D", "loss_cycle", "loss_identity", "batches_per_epoch"])

            for epoch_idx in range(len(history["loss_G"])):
                writer.writerow([
                    round_id,
                    epoch_idx + 1,
                    history["loss_G"][epoch_idx],
                    history["loss_D"][epoch_idx],
                    history["loss_cycle"][epoch_idx],
                    history["loss_identity"][epoch_idx],
                    history["batches_per_epoch"][epoch_idx],
                ])

        latest_metrics = {k: v[-1] for k, v in history.items()}
        return get_weights(self.net), len(train_loader.dataset), latest_metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        self.net = self.net.to(self.device)
        self.net.eval()

        test_horse_loader, test_zebra_loader = self.testloaders
        device = self.device
        l1 = nn.L1Loss()
        total_loss = 0.0
        count = 0

        generate_images = config.get("round") == config.get("num-server-rounds")
        save_dir = None
        if generate_images:
            save_dir = Path(f"./generated/client_{self.client_id}")
            save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx, (horse_img, zebra_img) in enumerate(zip(test_horse_loader, test_zebra_loader)):
                horse_img = horse_img.to(device)
                zebra_img = zebra_img.to(device)

                fake_zebra = self.net.G_AB(horse_img)
                fake_horse = self.net.G_BA(zebra_img)
                recov_horse = self.net.G_BA(fake_zebra)
                recov_zebra = self.net.G_AB(fake_horse)

                loss_cycle = l1(recov_horse, horse_img) + l1(recov_zebra, zebra_img)
                total_loss += loss_cycle.item()
                count += 1

                # === 仅最后一轮保存前 10 张图像 ===
                if generate_images and idx < 10:
                    from torchvision.utils import save_image
                    save_image(horse_img, save_dir / f"horse_{idx:03d}_real.png", normalize=True)
                    save_image(fake_zebra, save_dir / f"horse_{idx:03d}_fake_zebra.png", normalize=True)
                    save_image(zebra_img, save_dir / f"zebra_{idx:03d}_real.png", normalize=True)
                    save_image(fake_horse, save_dir / f"zebra_{idx:03d}_fake_horse.png", normalize=True)

        avg_loss = total_loss / count if count > 0 else 0.0
        return float(avg_loss), len(test_horse_loader.dataset), {"eval_loss": avg_loss}


    #
    # def evaluate(self, parameters, config):
    #     """Evaluate the model on the data this client has."""
    #     logger.info(f"Client {self.client_id} - Starting evaluation")
    #     set_weights(self.net, parameters)
    #     loss = test(self.net, self.testloader, self.device, client_id=self.client_id)
    #     logger.info(f"Client {self.client_id} - Evaluation completed with loss: {loss:.4f}")
    #     return float(loss), len(self.testloader), {"loss": float(loss)}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config.get("alpha", 0.5)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    paired_train_loader, (test_horse_loader, test_zebra_loader) = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
    )

    return CycleGANClient(
        trainloader=paired_train_loader,
        testloaders=(test_horse_loader, test_zebra_loader),
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        client_id=partition_id
    ).to_client()



app = ClientApp(client_fn=client_fn)
