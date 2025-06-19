"""fedvaeexample: A Flower / PyTorch app for Federated CycleGAN."""

from fedvaeexample.task import CycleGAN, get_weights

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAdam, FedProx
import json

loss_log = []

# def save_metrics(metrics):
#     # 每轮调用一次
#     round_num = len(loss_log) + 1
#     loss_log.append({"round": round_num, **metrics})
#     with open("federated_loss_log.json", "w") as f:
#         import json
#         json.dump(loss_log, f, indent=2)

# 添加聚合函数

def weighted_average(metrics):
    total_examples = sum(num for num, _ in metrics)
    avg_metrics = {
        k: sum(num * m[k] for num, m in metrics) / total_examples
        for k in metrics[0][1].keys()
    }

    # 📝 Save per round
    loss_log.append(avg_metrics)
    with open("results/round_loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    return avg_metrics

def fit_config_fn(server_round: int):
    return {"round": server_round}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    ndarrays = get_weights(CycleGAN())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    # ✅ 添加 evaluate_metrics_aggregation_fn

    # strategy = FedAvg(
    #     on_fit_config_fn=lambda rnd: {"round": rnd, "local_epochs": 2, "learning_rate": 0.0002},
    #     initial_parameters=parameters,
    #     evaluate_metrics_aggregation_fn=weighted_average
    # )

    # strategy = FedAdam(
    #     eta=0.001,
    #     beta_1=0.9,
    #     beta_2=0.99,
    #     tau=1e-3,
    #     min_fit_clients=2,
    #     min_available_clients=2,
    #     on_fit_config_fn=fit_config_fn,
    #     initial_parameters=parameters
    # )
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        proximal_mu=0.1,  # ← 关键参数！决定 FedProx 正则强度
        on_fit_config_fn=fit_config_fn,
        initial_parameters=parameters
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
