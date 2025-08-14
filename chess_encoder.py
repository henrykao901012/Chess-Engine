"""
西洋棋棋盤編碼器
負責將棋盤狀態和走法編碼為神經網路輸入/輸出
"""

import numpy as np
import chess
from typing import List, Optional


class ChessEncoder:
    """西洋棋盤面編碼器"""
    
    # 平面定義：6種棋子 × 2種顏色 = 12個平面
    PIECE_PLANES = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # 歷史平面數
    HISTORY_PLANES = 8  # 最近8個半步的盤面
    
    # 額外特徵平面
    EXTRA_PLANES = 7  # 顏色、總步數、王車易位權、過路兵、重複次數、無進展步數、50步規則
    
    # 總平面數
    TOTAL_PLANES = 12 * HISTORY_PLANES + EXTRA_PLANES  # 103個平面
    
    @staticmethod
    def encode_board(board: chess.Board, history: List[chess.Board] = None) -> np.ndarray:
        """
        將棋盤編碼為神經網路輸入
        返回: (103, 8, 8) 的張量
        """
        if history is None:
            history = []
        
        # 確保歷史長度為8
        full_history = history[-8:] if len(history) >= 8 else history + [board] * (8 - len(history))
        
        planes = []
        
        # 編碼歷史盤面
        for hist_board in full_history:
            # 己方棋子
            for piece_type in range(1, 7):
                plane = np.zeros((8, 8), dtype=np.float32)
                for square in hist_board.pieces(piece_type, hist_board.turn):
                    row, col = divmod(square, 8)
                    plane[7-row, col] = 1
                planes.append(plane)
            
            # 對方棋子
            for piece_type in range(1, 7):
                plane = np.zeros((8, 8), dtype=np.float32)
                for square in hist_board.pieces(piece_type, not hist_board.turn):
                    row, col = divmod(square, 8)
                    plane[7-row, col] = 1
                planes.append(plane)
        
        # 額外特徵
        # 當前回合顏色
        color_plane = np.full((8, 8), float(board.turn), dtype=np.float32)
        planes.append(color_plane)
        
        # 總步數（歸一化到0-1）
        move_count_plane = np.full((8, 8), min(board.fullmove_number / 100.0, 1.0), dtype=np.float32)
        planes.append(move_count_plane)
        
        # 王車易位權
        castling_planes = []
        for castling_right in [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]:
            plane = np.full((8, 8), float(bool(board.castling_rights & castling_right)), dtype=np.float32)
            castling_planes.append(plane)
        planes.extend(castling_planes)
        
        # 過路兵
        en_passant_plane = np.zeros((8, 8), dtype=np.float32)
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            en_passant_plane[7-row, col] = 1
        planes.append(en_passant_plane)
        
        return np.array(planes, dtype=np.float32)
    
    @staticmethod
    def encode_action(move: chess.Move) -> int:
        """
        將走法編碼為動作索引
        使用 8×8×73 的動作空間（Queen moves + Knight moves + Underpromotions）
        """
        from_square = move.from_square
        to_square = move.to_square
        
        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)
        
        # 計算方向和距離
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # Queen moves (56種): 8個方向 × 7個距離
        # Knight moves (8種)
        # Underpromotions (9種): 3種升變 × 3個方向
        
        # 基礎索引
        base_idx = from_row * 8 + from_col
        
        # 判斷移動類型
        if abs(row_diff) == 2 and abs(col_diff) == 1:
            # 騎士移動（垂直為主）
            knight_moves = {
                (2, 1): 56, (2, -1): 57,
                (-2, 1): 58, (-2, -1): 59
            }
            plane = knight_moves.get((row_diff, col_diff), 0)
        elif abs(row_diff) == 1 and abs(col_diff) == 2:
            # 騎士移動（水平為主）
            knight_moves = {
                (1, 2): 60, (1, -2): 61,
                (-1, 2): 62, (-1, -2): 63
            }
            plane = knight_moves.get((row_diff, col_diff), 0)
        elif move.promotion and move.promotion != chess.QUEEN:
            # 低升變
            promotion_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
            direction_offset = col_diff + 1  # -1, 0, 1 -> 0, 1, 2
            plane = 64 + promotion_offset[move.promotion] * 3 + direction_offset
        else:
            # Queen moves（包含普通升變為皇后）
            if row_diff == 0:  # 水平
                direction = 0 if col_diff > 0 else 1
                distance = abs(col_diff) - 1
            elif col_diff == 0:  # 垂直
                direction = 2 if row_diff > 0 else 3
                distance = abs(row_diff) - 1
            elif row_diff == col_diff:  # 主對角線
                direction = 4 if row_diff > 0 else 5
                distance = abs(row_diff) - 1
            else:  # 副對角線
                direction = 6 if row_diff > 0 else 7
                distance = abs(row_diff) - 1
            
            plane = direction * 7 + distance
        
        return base_idx * 73 + plane
    
    @staticmethod
    def decode_action(action_idx: int, board: chess.Board) -> Optional[chess.Move]:
        """將動作索引解碼為走法"""
        from_square = action_idx // 73
        plane = action_idx % 73
        
        from_row, from_col = divmod(from_square, 8)
        
        if plane < 56:  # Queen moves
            direction = plane // 7
            distance = (plane % 7) + 1
            
            # 方向向量
            directions = [
                (0, 1), (0, -1),   # 水平
                (1, 0), (-1, 0),   # 垂直
                (1, 1), (-1, -1),  # 主對角線
                (1, -1), (-1, 1)   # 副對角線
            ]
            
            row_dir, col_dir = directions[direction]
            to_row = from_row + row_dir * distance
            to_col = from_col + col_dir * distance
            
        elif plane < 64:  # Knight moves
            knight_moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            row_offset, col_offset = knight_moves[plane - 56]
            to_row = from_row + row_offset
            to_col = from_col + col_offset
            
        else:  # Underpromotions
            promotion_plane = plane - 64
            promotion_type = promotion_plane // 3
            direction = promotion_plane % 3 - 1  # 0,1,2 -> -1,0,1
            
            promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            promotion = promotions[promotion_type]
            
            to_row = from_row + (1 if board.turn else -1)
            to_col = from_col + direction
        
        # 檢查是否在棋盤內
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            return None
        
        to_square = to_row * 8 + to_col
        
        # 創建走法
        if plane >= 64:  # Underpromotion
            move = chess.Move(from_square, to_square, promotion=promotion)
        else:
            move = chess.Move(from_square, to_square)
            # 檢查是否需要升變為皇后
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if (board.turn and to_row == 7) or (not board.turn and to_row == 0):
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        # 驗證合法性
        if move in board.legal_moves:
            return move
        
        return None