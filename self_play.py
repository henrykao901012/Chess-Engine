"""
自對弈模組
負責執行並行自對弈遊戲生成訓練資料
"""

import time
import chess
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from typing import List
from .config import AlphaZeroConfig
from .model import AlphaZeroNet
from .mcts import MCTS
from .data_utils import GameRecord


class SelfPlayWorker:
    """自對弈工作器"""
    
    def __init__(self, worker_id: int, model: AlphaZeroNet, config: AlphaZeroConfig):
        self.worker_id = worker_id
        self.model = model
        self.config = config
        self.mcts = MCTS(model, config)
        
    def play_game(self) -> GameRecord:
        """執行一場自對弈"""
        board = chess.Board()
        game_record = GameRecord()
        game_record.metadata['worker_id'] = self.worker_id
        game_record.metadata['start_time'] = time.time()
        
        move_count = 0
        
        while not board.is_game_over() and move_count < self.config.max_game_length:
            # 決定溫度
            if move_count < self.config.temperature_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1  # 降低溫度以選擇最佳動作
            
            # MCTS搜索
            action_probs = self.mcts.search(
                board, 
                temperature=temperature,
                add_noise=(move_count < 30)  # 前30步添加噪音
            )
            
            # 選擇動作
            if temperature > 0:
                moves = list(action_probs.keys())
                probs = list(action_probs.values())
                action = np.random.choice(moves, p=probs)
            else:
                action = max(action_probs, key=action_probs.get)
            
            # 記錄
            game_record.add_move(board, action, action_probs)
            
            # 執行動作
            board.push(action)
            move_count += 1
            
            # 清理快取（保留子樹）
            if move_count % 10 == 0:
                self.mcts.clear_cache(keep_subtree=board)
        
        # 設置結果
        game_record.set_result(board.result())
        game_record.metadata['end_time'] = time.time()
        game_record.metadata['move_count'] = move_count
        
        return game_record


class ParallelSelfPlay:
    """並行自對弈協調器"""
    
    def __init__(self, model: AlphaZeroNet, config: AlphaZeroConfig):
        self.model = model
        self.config = config
        self.workers = []
        self.game_queue = Queue()
        
    def run_games(self, num_games: int) -> List[GameRecord]:
        """並行執行多場對弈"""
        games = []
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 提交任務
            futures = []
            for i in range(num_games):
                worker = SelfPlayWorker(i % self.config.num_workers, self.model, self.config)
                future = executor.submit(worker.play_game)
                futures.append(future)
            
            # 收集結果
            for future in futures:
                game = future.result()
                games.append(game)
                
                # 顯示進度
                if len(games) % 100 == 0:
                    logging.info(f"完成 {len(games)}/{num_games} 場對弈")
        
        return games