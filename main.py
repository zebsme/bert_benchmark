# benchmark.py
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TransformerModel(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.transformer(src)


class Benchmark:
    def __init__(
            self,
            model: nn.Module,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda",
    ):
        self.model = model.to(device).to(dtype)  # 确保模型参数也是 float16
        self.dtype = dtype
        self.device = device
        self.model.eval()

    def generate_dummy_data(
            self,
            batch_size: int,
            seq_len: int,
            d_model: int = 512,
    ) -> torch.Tensor:
        return torch.randn(
            seq_len,
            batch_size,
            d_model,
            dtype=self.dtype,
            device=self.device,
        )

    def measure_memory(self) -> float:
        """Measure GPU memory usage in MB"""
        return torch.cuda.memory_allocated() / 1024 / 1024

    def benchmark_eager(
            self,
            batch_size: int,
            seq_len: int,
            num_rounds: int = 10,
    ) -> Tuple[float, float]:
        # Warmup
        dummy_input = self.generate_dummy_data(batch_size, seq_len)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
        times = []
        for _ in range(num_rounds):
            dummy_input = self.generate_dummy_data(batch_size, seq_len)

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        avg_time = np.mean(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

        return avg_time, peak_memory

    def benchmark_sdpa(
            self,
            batch_size: int,
            seq_len: int,
            num_rounds: int = 10,
    ) -> Tuple[float, float]:
        # Enable scaled dot product attention
        torch.backends.cuda.enable_flash_sdp(True)

        # Similar to benchmark_eager
        dummy_input = self.generate_dummy_data(batch_size, seq_len)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(num_rounds):
            dummy_input = self.generate_dummy_data(batch_size, seq_len)

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        avg_time = np.mean(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Disable scaled dot product attention
        torch.backends.cuda.enable_flash_sdp(False)

        return avg_time, peak_memory


def run_benchmarks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize model and benchmark
    model = TransformerModel()
    benchmark = Benchmark(model)

    # Training configurations
    train_configs = [
        (4, 256), (4, 512),
        (8, 256), (8, 512),
        (16, 256), (16, 512),
    ]

    # Inference configurations
    infer_configs = [
        (1, 128), (1, 256),
        (2, 128), (2, 256),
        (4, 128), (4, 256),
    ]

    train_results = []
    infer_results = []

    # Run training benchmarks
    for batch_size, seq_len in train_configs:
        eager_time, eager_mem = benchmark.benchmark_eager(batch_size, seq_len)
        sdpa_time, sdpa_mem = benchmark.benchmark_sdpa(batch_size, seq_len)

        speedup = (eager_time - sdpa_time) / eager_time * 100
        mem_saving = (eager_mem - sdpa_mem) / eager_mem * 100

        train_results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'time_eager': eager_time,
            'time_sdpa': sdpa_time,
            'speedup': speedup,
            'mem_eager': eager_mem,
            'mem_sdpa': sdpa_mem,
            'mem_saving': mem_saving,
        })

    # Run inference benchmarks
    for batch_size, seq_len in infer_configs:
        eager_time, eager_mem = benchmark.benchmark_eager(batch_size, seq_len)
        sdpa_time, sdpa_mem = benchmark.benchmark_sdpa(batch_size, seq_len)

        # Convert to per-token latency (ms)
        eager_latency = eager_time / (batch_size * seq_len) * 1000
        sdpa_latency = sdpa_time / (batch_size * seq_len) * 1000
        speedup = (eager_latency - sdpa_latency) / eager_latency * 100
        mem_saved = (eager_mem - sdpa_mem) / eager_mem * 100

        infer_results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'latency_eager': eager_latency,
            'latency_sdpa': sdpa_latency,
            'speedup': speedup,
            'mem_eager': eager_mem,
            'mem_sdpa': sdpa_mem,
            'mem_saved': mem_saved,
        })

    return pd.DataFrame(train_results), pd.DataFrame(infer_results)


def plot_results(df_train: pd.DataFrame, df_infer: pd.DataFrame) -> None:
    plt.style.use('seaborn')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training time comparison
    width = 0.35
    x = np.arange(len(df_train))
    labels = [f'BS{b}_SL{s}' for b, s in zip(df_train['batch_size'], df_train['seq_len'])]

    ax1.bar(x - width / 2, df_train['time_eager'], width, label='Eager')
    ax1.bar(x + width / 2, df_train['time_sdpa'], width, label='SDPA')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('Time per batch (s)')
    ax1.legend()

    # Plot training memory usage
    ax2.bar(x - width / 2, df_train['mem_eager'], width, label='Eager')
    ax2.bar(x + width / 2, df_train['mem_sdpa'], width, label='SDPA')
    ax2.set_title('Training Memory Usage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()

    # Plot inference latency
    x_infer = np.arange(len(df_infer))
    labels_infer = [f'BS{b}_SL{s}' for b, s in zip(df_infer['batch_size'], df_infer['seq_len'])]

    ax3.bar(x_infer - width / 2, df_infer['latency_eager'], width, label='Eager')
    ax3.bar(x_infer + width / 2, df_infer['latency_sdpa'], width, label='SDPA')
    ax3.set_title('Inference Latency')
    ax3.set_xticks(x_infer)
    ax3.set_xticklabels(labels_infer, rotation=45)
    ax3.set_ylabel('Per token latency (ms)')
    ax3.legend()

    # Plot speedup comparison
    ax4.plot(x, df_train['speedup'], 'o-', label='Training')
    ax4.plot(x_infer, df_infer['speedup'], 's-', label='Inference')
    ax4.set_title('Speedup Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_ylabel('Speedup (%)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()


if __name__ == "__main__":
    # Run benchmarks and plot results
    df_train, df_infer = run_benchmarks()

    # Save results to CSV
    df_train.to_csv('training_results.csv', index=False)
    df_infer.to_csv('inference_results.csv', index=False)

    # Plot and save results
    plot_results(df_train, df_infer)

    print("Training Results:")
    print(df_train)
    print("\nInference Results:")
    print(df_infer)
