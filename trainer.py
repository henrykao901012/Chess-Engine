"""
AlphaZero 訓練器
負責整個訓練循環，包括自對弈、模型訓練和評估
"""

import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import logging
import numpy as np
import chess
from typing import List, Dict
from .config import AlphaZeroConfig
from .model import AlphaZeroNet
from .mcts import MCTS
from .data_utils import GameRecord, ReplayBuffer, DataAugmentation
from .self_play import ParallelSelfPlay
from .chess_encoder import ChessEncoder


class ChessDataset(Dataset):
    """西洋棋訓練資料集"""
    
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
    
    def __len__(self):
        return len(self.replay_buffer)
    
    def __getitem__(self, idx):
        # 這裡實際上不會用到idx，因為我們使用優先採樣
        return None  # 實際採樣在訓練循環中進行


class AlphaZeroTrainer:
    """AlphaZero 訓練器"""
    
    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 創建模型
        self.model = AlphaZeroNet(config).to(self.device)
        self.target_model = AlphaZeroNet(config).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 優化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_schedule_steps,
            gamma=config.lr_schedule_gamma
        )
        
        # 經驗回放緩衝區
        self.replay_buffer = ReplayBuffer(
            config.replay_buffer_size,
            alpha=config.prioritized_replay_alpha
        )
        
        # 自對弈模組
        self.self_play = ParallelSelfPlay(self.target_model, config)
        
        # 統計資訊
        self.stats = {
            'iteration': 0,
            'games_played': 0,
            'training_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'elo_rating': 1500,
            'win_rate': []
        }
        
        # 創建目錄
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('games', exist_ok=True)
        
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training.log'),
                logging.StreamHandler()
            ]
        )
    
    def train_iteration(self):
        """執行一次訓練迭代"""
        iteration = self.stats['iteration']
        logging.info(f"開始迭代 {iteration}")
        
        # 1. 自對弈生成資料
        logging.info("執行自對弈...")
        games = self.self_play.run_games(self.config.num_games_per_iteration)
        
        # 2. 處理遊戲資料
        logging.info("處理遊戲資料...")
        self._process_games(games)
        
        # 3. 訓練神經網路
        logging.info("訓練神經網路...")
        train_stats = self._train_network()
        
        # 4. 評估模型
        logging.info("評估模型...")
        should_update = self._evaluate_model()
        
        if should_update:
            logging.info("更新目標模型")
            self.target_model.load_state_dict(self.model.state_dict())
            self._update_elo(True)
        else:
            logging.info("保持當前目標模型")
            self._update_elo(False)
        
        # 5. 保存檢查點
        if iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
        
        # 6. 更新統計
        self.stats['iteration'] += 1
        self.stats['games_played'] += len(games)
        
        # 7. 記錄指標
        self._log_metrics(train_stats)
    
    def _process_games(self, games: List[GameRecord]):
        """處理遊戲資料並加入回放緩衝區"""
        for game in games:
            for i, (state, action, action_probs, reward) in enumerate(
                zip(game.states, game.actions, game.action_probs, game.rewards)
            ):
                # 編碼狀態
                state_tensor = ChessEncoder.encode_board(state)
                
                # 創建策略目標
                policy_target = np.zeros(8 * 8 * 73)
                for move, prob in action_probs.items():
                    action_idx = ChessEncoder.encode_action(move)
                    policy_target[action_idx] = prob
                
                # 資料增強
                augmented_samples = DataAugmentation.augment_chess_data(
                    state_tensor, policy_target, reward
                )
                
                # 加入緩衝區
                for aug_state, aug_policy, aug_value in augmented_samples:
                    sample = (aug_state, aug_policy, aug_value)
                    self.replay_buffer.push(sample)
    
    def _train_network(self) -> Dict:
        """訓練神經網路"""
        self.model.train()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 100  # 每次迭代的批次數
        
        beta = min(1.0, self.config.prioritized_replay_beta + 
                  self.stats['iteration'] * (1.0 - self.config.prioritized_replay_beta) / 1000)
        
        for batch_idx in range(num_batches):
            # 優先採樣
            samples, indices, weights = self.replay_buffer.sample(
                self.config.batch_size, beta
            )
            
            if len(samples) == 0:
                continue
            
            # 準備批次資料
            states = torch.FloatTensor([s[0] for s in samples]).to(self.device)
            policy_targets = torch.FloatTensor([s[1] for s in samples]).to(self.device)
            value_targets = torch.FloatTensor([s[2] for s in samples]).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # 前向傳播
            policy_preds, value_preds = self.model(states)
            
            # 計算損失
            policy_loss = -(policy_targets * F.log_softmax(policy_preds, dim=1)).sum(dim=1)
            value_loss = F.mse_loss(value_preds, value_targets, reduction='none').squeeze()
            
            # 加權損失
            policy_loss = (policy_loss * weights).mean()
            value_loss = (value_loss * weights).mean()
            
            loss = policy_loss + value_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新優先級
            with torch.no_grad():
                td_errors = torch.abs(value_preds - value_targets).squeeze().cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
            
            # 統計
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # 更新學習率
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _evaluate_model(self) -> bool:
        """評估新模型是否優於目標模型"""
        wins = 0
        draws = 0
        losses = 0
        
        for game_idx in range(self.config.evaluation_games):
            # 新模型執白
            if game_idx % 2 == 0:
                result = self._play_evaluation_game(self.model, self.target_model)
                if result == "1-0":
                    wins += 1
                elif result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1
            # 新模型執黑
            else:
                result = self._play_evaluation_game(self.target_model, self.model)
                if result == "0-1":
                    wins += 1
                elif result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1
        
        win_rate = (wins + 0.5 * draws) / self.config.evaluation_games
        self.stats['win_rate'].append(win_rate)
        
        logging.info(f"評估結果: 勝{wins} 和{draws} 負{losses} (勝率: {win_rate:.2%})")
        
        return win_rate >= self.config.evaluation_win_rate_threshold
    
    def _play_evaluation_game(self, white_model: AlphaZeroNet, black_model: AlphaZeroNet) -> str:
        """執行一場評估對局"""
        board = chess.Board()
        white_mcts = MCTS(white_model, self.config)
        black_mcts = MCTS(black_model, self.config)
        
        move_count = 0
        
        while not board.is_game_over() and move_count < self.config.max_game_length:
            if board.turn == chess.WHITE:
                mcts = white_mcts
            else:
                mcts = black_mcts
            
            # 評估時使用低溫度
            action_probs = mcts.search(board, num_simulations=400, temperature=0.1, add_noise=False)
            action = max(action_probs, key=action_probs.get)
            
            board.push(action)
            move_count += 1
        
        return board.result()
    
    def _update_elo(self, won: bool):
        """更新ELO評分"""
        k = 32  # ELO K因子
        expected = 1 / (1 + 10 ** ((1500 - self.stats['elo_rating']) / 400))
        actual = 1 if won else 0
        
        self.stats['elo_rating'] += k * (actual - expected)
    
    def _save_checkpoint(self):
        """保存檢查點"""
        checkpoint = {
            'iteration': self.stats['iteration'],
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stats': self.stats,
            'config': self.config
        }
        
        filename = f"checkpoints/checkpoint_iter_{self.stats['iteration']}.pt"
        torch.save(checkpoint, filename)
        logging.info(f"保存檢查點: {filename}")
        
        # 清理舊檢查點
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理舊的檢查點"""
        checkpoints = sorted([
            f for f in os.listdir('checkpoints') 
            if f.startswith('checkpoint_')
        ])
        
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                os.remove(os.path.join('checkpoints', checkpoint))
    
    def _log_metrics(self, train_stats: Dict):
        """記錄訓練指標"""
        self.stats['training_loss'].append(train_stats['loss'])
        self.stats['policy_loss'].append(train_stats['policy_loss'])
        self.stats['value_loss'].append(train_stats['value_loss'])
        
        # 寫入JSON
        with open('logs/metrics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # 記錄到日誌
        logging.info(f"迭代 {self.stats['iteration']} - "
                    f"損失: {train_stats['loss']:.4f} "
                    f"(策略: {train_stats['policy_loss']:.4f}, "
                    f"價值: {train_stats['value_loss']:.4f}) "
                    f"ELO: {self.stats['elo_rating']:.0f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入檢查點"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.stats = checkpoint['stats']
        
        logging.info(f"載入檢查點: {checkpoint_path}")