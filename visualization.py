"""
視覺化工具模組
用於顯示訓練進度和MCTS搜索樹
"""

import chess
import chess.pgn
from datetime import datetime
from .trainer import AlphaZeroTrainer
from .mcts import MCTSNode
from .data_utils import GameRecord


class AlphaZeroVisualizer:
    """視覺化工具"""
    
    def __init__(self, trainer: AlphaZeroTrainer):
        self.trainer = trainer
        
    def plot_training_curves(self):
        """繪製訓練曲線"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安裝 matplotlib 來繪製圖表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失曲線
        axes[0, 0].plot(self.trainer.stats['training_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        # 策略損失
        axes[0, 1].plot(self.trainer.stats['policy_loss'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        
        # 價值損失
        axes[1, 0].plot(self.trainer.stats['value_loss'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        
        # ELO評分
        axes[1, 1].plot(range(len(self.trainer.stats['win_rate'])), 
                       [self.trainer.stats['elo_rating']] * len(self.trainer.stats['win_rate']))
        axes[1, 1].set_title('ELO Rating')
        axes[1, 1].set_xlabel('Evaluation')
        axes[1, 1].set_ylabel('ELO')
        
        plt.tight_layout()
        plt.savefig('logs/training_curves.png')
        plt.show()
    
    def visualize_mcts_tree(self, root: MCTSNode, depth: int = 3):
        """視覺化MCTS搜索樹"""
        try:
            import graphviz
        except ImportError:
            print("需要安裝 graphviz 來繪製搜索樹")
            return
        
        dot = graphviz.Digraph(comment='MCTS Tree')
        
        def add_node(node: MCTSNode, parent_id: str = None, depth_remaining: int = depth):
            if depth_remaining <= 0:
                return
            
            node_id = str(id(node))
            label = f"V:{node.value():.2f}\\nN:{node.visits}"
            
            dot.node(node_id, label)
            
            if parent_id:
                dot.edge(parent_id, node_id)
            
            # 添加子節點（只顯示訪問次數最多的前5個）
            sorted_children = sorted(node.children.values(), 
                                   key=lambda n: n.visits, 
                                   reverse=True)[:5]
            
            for child in sorted_children:
                add_node(child, node_id, depth_remaining - 1)
        
        add_node(root)
        
        dot.render('logs/mcts_tree', format='png', cleanup=True)
    
    def create_game_replay(self, game_record: GameRecord) -> str:
        """創建遊戲回放HTML"""
        pgn = chess.pgn.Game()
        node = pgn
        
        for move in game_record.actions:
            node = node.add_variation(move)
        
        pgn.headers["Result"] = game_record.game_result
        pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        return str(pgn)