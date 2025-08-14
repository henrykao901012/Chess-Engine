"""
AlphaZero 配置模組
定義訓練和模型的所有超參數
"""

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class AlphaZeroConfig:
    """AlphaZero 訓練配置"""
    # 神經網路參數
    num_res_blocks: int = 19
    num_filters: int = 256
    
    # MCTS 參數
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    
    # 訓練參數
    batch_size: int = 256
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9
    lr_schedule_steps: List[int] = None
    lr_schedule_gamma: float = 0.1
    
    # 自對弈參數
    num_parallel_games: int = 100
    num_games_per_iteration: int = 5000
    max_game_length: int = 512
    
    # 經驗回放參數
    replay_buffer_size: int = 500000
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4
    
    # 模型評估參數
    evaluation_games: int = 100
    evaluation_win_rate_threshold: float = 0.55
    
    # 儲存參數
    checkpoint_interval: int = 100
    keep_last_n_checkpoints: int = 5
    
    # 資源管理
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    def __post_init__(self):
        if self.lr_schedule_steps is None:
            self.lr_schedule_steps = [100000, 200000, 300000]