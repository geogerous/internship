#!/usr/bin/env python3
"""
MRPNN 训练脚本 - 包含方差特征的改进版本

此脚本用于训练 MRPNN 神经网络，支持：
1. 从 CSV 文件加载训练数据
2. 使用方差特征作为额外输入通道
3. 实现论文中定义的对数均方误差损失函数
4. 使用 Adabound 优化器进行训练
5. 支持加权损失函数（对碎片化区域赋予更高权重）
6. 自动保存最佳模型权重为 MinLossWeights.txt 格式

作者: Manus AI
日期: 2026-02-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from typing import Tuple, Optional

# 尝试导入 Adabound 优化器
try:
    from adabound import AdaBound
    ADABOUND_AVAILABLE = True
except ImportError:
    print("警告: adabound 未安装，将使用 Adam 优化器")
    print("安装方法: pip install adabound")
    ADABOUND_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MRPNNDataset(Dataset):
    """
    MRPNN 数据集类
    
    从 dataGen.cu 生成的 CSV 文件中加载训练数据。
    CSV 格式：
    - 前 N 列：描述符特征（密度、透射率、相位、方差等）
    - 倒数第 2 列：Gamma（视角与光照方向的夹角）
    - 最后 1 列：Radiance（真值）
    """
    
    def __init__(self, csv_path: str, use_variance: bool = True):
        """
        Args:
            csv_path: CSV 文件路径
            use_variance: 是否使用方差特征
        """
        logger.info(f"加载数据集: {csv_path}")
        
        # 读取 CSV 文件（跳过注释行）
        self.data = pd.read_csv(csv_path, comment='#', header=None)
        
        logger.info(f"数据集大小: {len(self.data)} 个样本")
        logger.info(f"特征维度: {self.data.shape[1] - 1} (不含标签)")
        
        # 分离特征和标签
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
        
        # Gamma 是倒数第二列
        self.gamma = self.features[:, -1:]
        
        # 描述符特征是除了最后一列（Gamma）之外的所有列
        self.descriptors = self.features[:, :-1]
        
        self.use_variance = use_variance
        
        # 数据统计
        logger.info(f"Radiance 范围: [{self.labels.min():.4f}, {self.labels.max():.4f}]")
        logger.info(f"Gamma 范围: [{self.gamma.min():.4f}, {self.gamma.max():.4f}]")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            features: 输入特征 (描述符 + Gamma)
            label: 真值 Radiance
        """
        features = torch.from_numpy(self.features[idx])
        label = torch.from_numpy(self.labels[idx])
        
        return features, label


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module
    
    论文中使用的通道注意力机制，用于自适应地调整特征通道的权重。
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MRPNNNetwork(nn.Module):
    """
    MRPNN 神经网络
    
    基于论文 "Deep Real-time Volumetric Rendering Using Multi-feature Fusion" 的网络架构。
    支持方差特征作为额外输入通道。
    """
    
    def __init__(
        self,
        num_samples: int = 192,
        num_features_per_sample: int = 3,  # 密度、透射率、相位
        use_variance: bool = True,
        hidden_dim: int = 64,
        use_se: bool = True
    ):
        """
        Args:
            num_samples: 每个描述符的采样点数量（默认 192）
            num_features_per_sample: 每个采样点的特征数（3 或 4，取决于是否使用方差）
            use_variance: 是否使用方差特征
            hidden_dim: 隐藏层维度
            use_se: 是否使用 SE Module
        """
        super(MRPNNNetwork, self).__init__()
        
        self.num_samples = num_samples
        self.use_variance = use_variance
        self.use_se = use_se
        
        # 如果使用方差，每个采样点有 4 个特征（密度、透射率、相位、方差）
        if use_variance:
            num_features_per_sample = 4
        
        # 输入维度 = 采样点数 × 每点特征数 + 1（Gamma）
        input_dim = num_samples * num_features_per_sample + 1
        
        logger.info(f"网络输入维度: {input_dim}")
        logger.info(f"使用方差特征: {use_variance}")
        logger.info(f"使用 SE Module: {use_se}")
        
        # Feature Processing Stage
        self.feature_stage = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        if self.use_se:
            self.se_module = SEModule(hidden_dim * 2)
        
        # Albedo Stage (维度缩减)
        self.albedo_stage = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
        
        Returns:
            predicted_radiance: 预测的 Radiance [batch_size, 1]
        """
        # Feature Stage
        x = self.feature_stage(x)
        
        # SE Module (如果启用)
        if self.use_se:
            x = x.unsqueeze(2)  # [B, C, 1]
            x = self.se_module(x)
            x = x.squeeze(2)  # [B, C]
        
        # Albedo Stage
        x = self.albedo_stage(x)
        
        # 确保输出为正值
        x = F.relu(x)
        
        return x


class LogMSELoss(nn.Module):
    """
    对数均方误差损失函数
    
    论文中定义的损失函数：
    L = (log(Predicted / Albedo + 1) - log(GroundTruth / Albedo + 1))^2
    
    为简化，这里假设 Albedo = 1.0（或已经在数据预处理中处理）
    """
    
    def __init__(self, alpha: float = 4.0):
        """
        Args:
            alpha: Albedo 的指数参数（论文推荐 α = 4）
        """
        super(LogMSELoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [batch_size, 1]
            target: 真值 [batch_size, 1]
        
        Returns:
            loss: 标量损失值
        """
        # 对数变换
        pred_log = torch.log(pred + 1.0)
        target_log = torch.log(target + 1.0)
        
        # MSE
        loss = F.mse_loss(pred_log, target_log)
        
        return loss


class WeightedLogMSELoss(nn.Module):
    """
    加权对数均方误差损失函数
    
    对碎片化区域（高方差区域）赋予更高的权重。
    """
    
    def __init__(self, alpha: float = 4.0, weight_k: float = 2.0, use_variance: bool = True):
        """
        Args:
            alpha: Albedo 的指数参数
            weight_k: 方差权重系数
            use_variance: 是否使用方差加权
        """
        super(WeightedLogMSELoss, self).__init__()
        self.alpha = alpha
        self.weight_k = weight_k
        self.use_variance = use_variance
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [batch_size, 1]
            target: 真值 [batch_size, 1]
            features: 输入特征 [batch_size, feature_dim]（用于提取方差信息）
        
        Returns:
            loss: 标量损失值
        """
        # 对数变换
        pred_log = torch.log(pred + 1.0)
        target_log = torch.log(target + 1.0)
        
        # 计算未加权的损失
        unweighted_loss = (pred_log - target_log) ** 2
        
        if self.use_variance and features is not None:
            # 提取方差特征（假设方差特征在输入的特定位置）
            # 这里需要根据实际的特征排列进行调整
            # 假设每个采样点的第 4 个特征是方差/CV
            num_samples = 192
            variance_features = features[:, 3::4]  # 步长为 4，从第 3 个索引开始
            
            # 计算每个样本的平均方差度量
            sample_variance_metric = torch.mean(variance_features, dim=1, keepdim=True)
            
            # 计算权重
            weight = 1.0 + self.weight_k * sample_variance_metric
            
            # 加权损失
            weighted_loss = unweighted_loss * weight
            loss = torch.mean(weighted_loss)
        else:
            loss = torch.mean(unweighted_loss)
        
        return loss


def save_weights_to_txt(model: nn.Module, save_path: str):
    """
    将 PyTorch 模型权重保存为 MinLossWeights.txt 格式
    
    该格式可以被 tools/CastWeight.py 读取并转换为 C++ 头文件。
    
    Args:
        model: 训练好的模型
        save_path: 保存路径
    """
    logger.info(f"保存权重到: {save_path}")
    
    with open(save_path, 'w') as f:
        for name, param in model.named_parameters():
            # 将参数名转换为 C++ 风格
            cpp_name = name.replace('.', '_').replace('weight', 'W').replace('bias', 'B')
            
            # 获取参数数据
            data = param.data.cpu().numpy().flatten()
            
            # 写入文件
            f.write(f"const float {cpp_name}[{len(data)}] = {{\n")
            
            # 每行写入 10 个数值
            for i in range(0, len(data), 10):
                line_data = data[i:i+10]
                line_str = ', '.join([f"{val:.8f}" for val in line_data])
                f.write(f"    {line_str}")
                if i + 10 < len(data):
                    f.write(',\n')
                else:
                    f.write('\n')
            
            f.write('};\n\n')
    
    logger.info("权重保存完成")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_weighted_loss: bool = False
) -> float:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        use_weighted_loss: 是否使用加权损失
    
    Returns:
        average_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    
    with tqdm(dataloader, desc="训练中") as pbar:
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(features)
            
            # 计算损失
            if use_weighted_loss and isinstance(criterion, WeightedLogMSELoss):
                loss = criterion(outputs, labels, features)
            else:
                loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        average_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="验证中") as pbar:
            for features, labels in pbar:
                features = features.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(features)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='MRPNN 训练脚本')
    
    # 数据参数
    parser.add_argument('--train_csv', type=str, required=True,
                        help='训练数据 CSV 文件路径')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='验证数据 CSV 文件路径')
    
    # 模型参数
    parser.add_argument('--use_variance', action='store_true', default=True,
                        help='是否使用方差特征')
    parser.add_argument('--use_se', action='store_true', default=True,
                        help='是否使用 SE Module')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--use_weighted_loss', action='store_true', default=False,
                        help='是否使用加权损失函数')
    parser.add_argument('--weight_k', type=float, default=2.0,
                        help='方差权重系数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    train_dataset = MRPNNDataset(args.train_csv, use_variance=args.use_variance)
    val_dataset = MRPNNDataset(args.val_csv, use_variance=args.use_variance)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = MRPNNNetwork(
        num_samples=192,
        use_variance=args.use_variance,
        hidden_dim=args.hidden_dim,
        use_se=args.use_se
    ).to(device)
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义损失函数
    if args.use_weighted_loss:
        criterion = WeightedLogMSELoss(
            alpha=4.0,
            weight_k=args.weight_k,
            use_variance=args.use_variance
        )
        logger.info(f"使用加权损失函数，权重系数 k={args.weight_k}")
    else:
        criterion = LogMSELoss(alpha=4.0)
        logger.info("使用标准对数 MSE 损失函数")
    
    # 定义优化器
    if ADABOUND_AVAILABLE:
        optimizer = AdaBound(
            model.parameters(),
            lr=args.lr,
            final_lr=0.1
        )
        logger.info("使用 AdaBound 优化器")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("使用 Adam 优化器")
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 50)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_weighted_loss=args.use_weighted_loss
        )
        logger.info(f"训练损失: {train_loss:.6f}")
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"验证损失: {val_loss:.6f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"✓ 新的最佳模型！验证损失: {val_loss:.6f}")
            
            # 保存 PyTorch 模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            
            # 保存为 MinLossWeights.txt 格式
            save_weights_to_txt(model, save_dir / 'MinLossWeights.txt')
    
    logger.info("\n训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"模型已保存到: {save_dir}")
    logger.info(f"MinLossWeights.txt 已生成，可以使用 tools/CastWeight.py 转换为 C++ 头文件")


if __name__ == '__main__':
    main()