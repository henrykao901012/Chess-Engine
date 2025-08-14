"""
蒙特卡洛樹搜索 (MCTS) 實現
用於 AlphaZero 的決策樹搜索
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import chess
from typing import Dict, Optional
from .config import AlphaZeroConfig
from .chess_encoder import ChessEncoder


class MCTSNode:
    """MCTS 節點"""
    
    def __init__(self, state: chess.Board, parent=None, action=None, prior=0):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.visits = 0
        self.value_sum = 0
        self.children = {}
        self.is_expanded = False
        
        # 虛擬損失（用於並行MCTS）
        self.virtual_loss = 0
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct):
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + exploration
    
    def select_child(self, c_puct):
        """選擇UCB分數最高的子節點"""
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))
    
    def expand(self, priors):
        """擴展節點"""
        self.is_expanded = True
        for move in self.state.legal_moves:
            action_idx = ChessEncoder.encode_action(move)
            prior = priors[action_idx]
            child_state = self.state.copy()
            child_state.push(move)
            self.children[move] = MCTSNode(child_state, self, move, prior)
    
    def backup(self, value):
        """回傳價值"""
        self.visits += 1
        self.value_sum += value
        self.virtual_loss = 0
        
        if self.parent:
            self.parent.backup(-value)  # 對手視角，價值取負


class MCTS:
    """蒙特卡洛樹搜索"""
    
    def __init__(self, model, config: AlphaZeroConfig):
        self.model = model
        self.config = config
        self.cache = {}  # 搜索樹快取
    
    @torch.no_grad()
    def search(self, root_state: chess.Board, num_simulations: int = None, 
               temperature: float = 1.0, add_noise: bool = True) -> Dict[chess.Move, float]:
        """
        執行MCTS搜索
        返回: 動作訪問分布
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        # 檢查快取
        state_hash = root_state.fen()
        if state_hash in self.cache:
            root = self.cache[state_hash]
        else:
            root = MCTSNode(root_state)
            self.cache[state_hash] = root
        
        # 添加 Dirichlet噪音到根節點（用於探索）
        if add_noise and not root.is_expanded:
            self._expand_node(root, add_noise=True)
        
        # 執行模擬
        for _ in range(num_simulations):
            node = root
            path = [node]
            
            # 選擇
            while node.is_expanded and node.children:
                node = node.select_child(self.config.c_puct)
                path.append(node)
            
            # 擴展和評估
            if not node.state.is_game_over():
                if not node.is_expanded:
                    value = self._expand_node(node)
                else:
                    # 已擴展但沒有子節點（所有走法都不合法）
                    value = self._evaluate(node.state)
            else:
                # 遊戲結束
                result = node.state.result()
                if result == "1-0":
                    value = 1 if node.state.turn == chess.WHITE else -1
                elif result == "0-1":
                    value = -1 if node.state.turn == chess.WHITE else 1
                else:
                    value = 0
            
            # 回傳
            for n in reversed(path):
                n.backup(value)
                value = -value
        
        # 計算訪問分布
        visits = {move: child.visits for move, child in root.children.items()}
        
        if temperature == 0:
            # 選擇最佳動作
            best_move = max(visits, key=visits.get)
            action_probs = {move: 1.0 if move == best_move else 0.0 for move in visits}
        else:
            # 根據訪問次數計算概率
            total_visits = sum(visits.values())
            if total_visits > 0:
                action_probs = {
                    move: (count / total_visits) ** (1 / temperature)
                    for move, count in visits.items()
                }
                # 歸一化
                prob_sum = sum(action_probs.values())
                action_probs = {move: prob / prob_sum for move, prob in action_probs.items()}
            else:
                action_probs = {move: 1.0 / len(visits) for move in visits}
        
        return action_probs
    
    def _expand_node(self, node: MCTSNode, add_noise: bool = False) -> float:
        """擴展節點並返回價值評估"""
        # 編碼狀態
        state_tensor = ChessEncoder.encode_board(node.state)
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.config.device)
        
        # 神經網路評估
        self.model.eval()
        policy_logits, value = self.model(state_tensor)
        
        # 處理策略
        policy_logits = policy_logits.cpu().numpy()[0]
        priors = F.softmax(torch.FloatTensor(policy_logits), dim=0).numpy()
        
        # 過濾合法動作
        legal_actions = {}
        for move in node.state.legal_moves:
            action_idx = ChessEncoder.encode_action(move)
            legal_actions[action_idx] = move
        
        # 重新歸一化先驗概率
        legal_priors = {idx: priors[idx] for idx in legal_actions}
        prior_sum = sum(legal_priors.values())
        if prior_sum > 0:
            legal_priors = {idx: p / prior_sum for idx, p in legal_priors.items()}
        else:
            legal_priors = {idx: 1.0 / len(legal_actions) for idx in legal_actions}
        
        # 添加 Dirichlet噪音（根節點）
        if add_noise:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_priors))
            for i, idx in enumerate(legal_priors):
                legal_priors[idx] = (1 - self.config.dirichlet_epsilon) * legal_priors[idx] + \
                                   self.config.dirichlet_epsilon * noise[i]
        
        # 創建子節點
        node.is_expanded = True
        for action_idx, move in legal_actions.items():
            prior = legal_priors[action_idx]
            child_state = node.state.copy()
            child_state.push(move)
            node.children[move] = MCTSNode(child_state, node, move, prior)
        
        return value.item()
    
    def _evaluate(self, state: chess.Board) -> float:
        """評估終局狀態"""
        result = state.result()
        if result == "1-0":
            return 1 if state.turn == chess.WHITE else -1
        elif result == "0-1":
            return -1 if state.turn == chess.WHITE else 1
        else:
            return 0
    
    def clear_cache(self, keep_subtree: Optional[chess.Board] = None):
        """清理快取"""
        if keep_subtree:
            # 保留子樹
            new_cache = {}
            state_hash = keep_subtree.fen()
            if state_hash in self.cache:
                new_cache[state_hash] = self.cache[state_hash]
            self.cache = new_cache
        else:
            self.cache.clear()