"""
AlphaZero 西洋棋 AI 主程序
包含訓練循環和模型推理
"""

import logging
from config import AlphaZeroConfig
from trainer import AlphaZeroTrainer
from visualization import AlphaZeroVisualizer


def main():
    """主訓練程序"""
    # 創建配置
    config = AlphaZeroConfig(
        num_res_blocks=19,
        num_filters=256,
        num_simulations=800,
        batch_size=256,
        learning_rate=0.01,
        num_parallel_games=100,
        num_games_per_iteration=1000,
        replay_buffer_size=500000,
        evaluation_games=100,
        checkpoint_interval=10
    )
    
    # 創建訓練器
    trainer = AlphaZeroTrainer(config)
    
    # 可選：載入檢查點
    # trainer.load_checkpoint('checkpoints/checkpoint_iter_100.pt')
    
    # 創建視覺化工具
    visualizer = AlphaZeroVisualizer(trainer)
    
    # 訓練循環
    num_iterations = 1000
    
    for iteration in range(num_iterations):
        try:
            # 執行訓練迭代
            trainer.train_iteration()
            
            # 定期視覺化
            if iteration % 10 == 0:
                visualizer.plot_training_curves()
            
        except KeyboardInterrupt:
            logging.info("訓練被用戶中斷")
            break
        except Exception as e:
            logging.error(f"訓練錯誤: {e}")
            trainer._save_checkpoint()  # 緊急保存
            raise
    
    # 最終保存
    trainer._save_checkpoint()
    visualizer.plot_training_curves()
    
    logging.info("訓練完成！")


if __name__ == "__main__":
    main()