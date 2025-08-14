"""
資料處理工具模組
包含遊戲記錄、經驗回放緩衝區和資料增強
"""

import numpy as np
import chess
from collections import deque
from typing import List, Tuple, Dict
import time


class GameRecord:
    """遊戲記錄"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.game_result = None
        self.metadata = {}
    
    def add_move(self, state: chess.Board, action: chess.Move, action_prob: Dict[chess.Move, float]):
        self.states.append(state.copy())
        self.actions.append(action)
        self.action_probs.append(action_prob)
    
    def set_result(self, result: str):
        """設置遊戲結果"""
        self.game_result = result
        
        # 計算獎勵
        if result == "1-0":
            # 白方勝
            self.rewards = [1 if i % 2 == 0 else -1 for i in range(len(self.states))]
        elif result == "0-1":
            # 黑方勝
            self.rewards = [-1 if i % 2 == 0 else 1 for i in range(len(self.states))]
        else:
            # 和局
            self.rewards = [0] * len(self.states)


class ReplayBuffer:
    """經驗回放緩衝區（支持優先回放）"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, sample, priority: float = None):
        """添加樣本"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(sample)
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
        """優先採樣"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # 計算採樣概率
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # 採樣索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 計算重要性權重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # 獲取樣本
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新優先級"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class DataAugmentation:
    """資料增強（利用棋盤對稱性）"""
    
    @staticmethod
    def augment_chess_data(state: np.ndarray, policy: np.ndarray, value: float) -> List[Tuple]:
        """
        西洋棋的資料增強
        注意：西洋棋只有左右鏡像對稱（不能旋轉）
        """
        augmented_data = []
        
        # 原始資料
        augmented_data.append((state, policy, value))
        
        # 左右鏡像
        mirrored_state = DataAugmentation._mirror_state(state)
        mirrored_policy = DataAugmentation._mirror_policy(policy)
        augmented_data.append((mirrored_state, mirrored_policy, value))
        
        return augmented_data
    
    @staticmethod
    def _mirror_state(state: np.ndarray) -> np.ndarray:
        """鏡像棋盤狀態"""
        return np.flip(state, axis=2)  # 沿著列軸翻轉
    
    @staticmethod
    def _mirror_policy(policy: np.ndarray) -> np.ndarray:
        """鏡像策略向量"""
        # 重塑為8×8×73
        policy_3d = policy.reshape(8, 8, 73)
        
        # 鏡像每個平面
        mirrored = np.flip(policy_3d, axis=1)
        
        # 調整某些動作的方向
        # Queen moves的水平方向需要交換
        mirrored[:, :, 0:7], mirrored[:, :, 7:14] = \
            mirrored[:, :, 7:14].copy(), mirrored[:, :, 0:7].copy()
        
        # 對角線方向也需要調整
        mirrored[:, :, 35:42], mirrored[:, :, 42:49] = \
            mirrored[:, :, 42:49].copy(), mirrored[:, :, 35:42].copy()
        
        # Knight moves需要鏡像
        knight_mirror = {
            56: 57, 57: 56,  # 垂直騎士移動
            58: 59, 59: 58,
            60: 61, 61: 60,  # 水平騎士移動
            62: 63, 63: 62
        }
        
        for orig, mirror in knight_mirror.items():
            temp = mirrored[:, :, orig].copy()
            mirrored[:, :, orig] = mirrored[:, :, mirror]
            mirrored[:, :, mirror] = temp
        
        return mirrored.reshape(-1)