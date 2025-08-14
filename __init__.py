
# 核心配置
from .config import AlphaZeroConfig

# 棋盤編碼
from .chess_encoder import ChessEncoder

# 神經網路模型
from .model import AlphaZeroNet, ResidualBlock

# 蒙特卡洛樹搜索
from .mcts import MCTS, MCTSNode

# 資料處理工具
from .data_utils import GameRecord, ReplayBuffer, DataAugmentation

# 自對弈模組
from .self_play import SelfPlayWorker, ParallelSelfPlay

# 訓練器
from .trainer import AlphaZeroTrainer, ChessDataset

# 視覺化工具
from .visualization import AlphaZeroVisualizer

# 版本資訊
__version__ = "1.0.0"
__author__ = "AlphaZero Chess Team"
__email__ = "team@alphazero-chess.com"
__description__ = "基於 PyTorch 的西洋棋 AI 訓練框架，實現 AlphaZero 演算法"

# 公開 API
__all__ = [
    # 配置
    'AlphaZeroConfig',
    
    # 編碼器
    'ChessEncoder',
    
    # 模型
    'AlphaZeroNet',
    'ResidualBlock',
    
    # MCTS
    'MCTS',
    'MCTSNode',
    
    # 資料處理
    'GameRecord',
    'ReplayBuffer',
    'DataAugmentation',
    
    # 自對弈
    'SelfPlayWorker',
    'ParallelSelfPlay',
    
    # 訓練
    'AlphaZeroTrainer',
    'ChessDataset',
    
    # 視覺化
    'AlphaZeroVisualizer',
]

# 版本檢查
def check_dependencies():
    """檢查必要的依賴套件"""
    required_packages = [
        'torch',
        'chess', 
        'numpy',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"缺少必要的依賴套件: {', '.join(missing_packages)}\n"
            f"請執行: pip install {' '.join(missing_packages)}"
        )

# 在匯入時檢查依賴
try:
    check_dependencies()
except ImportError as e:
    import warnings
    warnings.warn(str(e), ImportWarning)