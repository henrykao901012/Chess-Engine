"""
AlphaZero 神經網路模型
包含殘差網路架構的策略-價值網路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AlphaZeroConfig
from .chess_encoder import ChessEncoder


class ResidualBlock(nn.Module):
    """殘差塊"""
    
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNet(nn.Module):
    """AlphaZero 神經網路"""
    
    def __init__(self, config: AlphaZeroConfig):
        super().__init__()
        self.config = config
        
        # 輸入層
        self.conv_input = nn.Conv2d(ChessEncoder.TOTAL_PLANES, config.num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(config.num_filters)
        
        # 殘差塔
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters) for _ in range(config.num_res_blocks)
        ])
        
        # 策略頭
        self.policy_conv = nn.Conv2d(config.num_filters, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 8 * 8 * 73)
        
        # 價值頭
        self.value_conv = nn.Conv2d(config.num_filters, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # 輸入處理
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 殘差塔
        for block in self.residual_blocks:
            x = block(x)
        
        # 策略輸出
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # 價值輸出
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value