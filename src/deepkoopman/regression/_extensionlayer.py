import torch
import torch.nn as nn
from typing import Dict, Optional #, Tuple

class CNNExtensionLayer(nn.Module):
    """
    CNN-based feature extension layer for extracting local features from feature vectors.
    
    Parameters
    ----------
    input_size : int
        Input feature dimension
    output_size : int
        Output feature dimension
    hidden_channels : list, default=[32, 64, 128]
        Number of channels for each convolutional layer
    kernel_sizes : list, default=[3, 5, 7]
        List of kernel sizes for multi-scale feature extraction
    dropout_rate : float, default=0.1
        Dropout rate
    use_residual : bool, default=True
        Whether to use residual connections
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_channels: list = [32, 64, 128],
        kernel_sizes: list = [3, 5, 7],
        dropout_rate: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_residual = use_residual
        
        # Build multi-scale convolutional networks
        self.conv_branches = nn.ModuleList()
        
        # Create a branch for each kernel size
        for kernel_size in kernel_sizes:
            branch = []
            in_channels = 1  # Only one input channel
            
            # Add convolutional layers
            for i, out_channels in enumerate(hidden_channels):
                padding = kernel_size // 2  # Use same padding to keep sequence length
                branch.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
                branch.append(nn.BatchNorm1d(out_channels))
                branch.append(nn.SiLU())
                
                if i < len(hidden_channels) - 1:  # Do not add dropout on the last layer
                    branch.append(nn.Dropout(dropout_rate))
                
                in_channels = out_channels
            
            self.conv_branches.append(nn.Sequential(*branch))
        
        # Compute the total output channels of the convolutional layers
        total_channels = sum(hidden_channels[-1] for _ in kernel_sizes)
        
        # Use adaptive pooling to reduce feature dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Add final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(total_channels, output_size),
            nn.LayerNorm(output_size),
            nn.SiLU()
        )
        
        # If using residual connection, add input projection layer
        if self.use_residual and input_size <= output_size:
            self.input_proj = nn.Linear(input_size, output_size)
            self.has_direct_skip = True
        else:
            self.has_direct_skip = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the CNN extension layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_size)
            
        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, output_size)
        """
        # Save input for residual connection
        identity = x
        
        # Reshape input for convolution (batch_size, channels, length)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Pass through each convolutional branch
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # (batch_size, out_channels, input_size)
            # Apply global pooling to reduce dimension
            branch_out = self.pool(branch_out).squeeze(-1)  # (batch_size, out_channels)
            branch_outputs.append(branch_out)
        
        # Concatenate outputs from all branches
        combined = torch.cat(branch_outputs, dim=1)  # (batch_size, total_channels)
        
        # Pass through fully connected layer to get final output
        output = self.fc(combined)  # (batch_size, output_size)
        
        # Add residual connection if applicable
        if self.use_residual and self.has_direct_skip:
            output = output + self.input_proj(identity)
        
        return output
    
class KernelExtensionLayer(nn.Module):
    """
    基于核函数的扩展层，用于捕获数据的局部特性
    
    Parameters
    ----------
    input_size : int
        输入特征维度
    output_size : int
        输出特征维度
    kernel_type : str, default="rbf"
        核函数类型: "rbf", "polynomial", "laplacian", "mixed"
    kernel_params : dict, default=None
        核函数的参数
    n_centers : int, default=50
        使用的核中心点数量
    learn_centers : bool, default=True
        是否学习中心点或随机初始化
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        kernel_type: str = "rbf", 
        kernel_params: Optional[Dict] = None, 
        n_centers: int = 50, 
        learn_centers: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_type = kernel_type
        self.n_centers = n_centers
        self.learn_centers = learn_centers
        
        # 设置默认核参数
        if kernel_params is None:
            if kernel_type == "rbf":
                self.kernel_params = {"gamma": 0.1}
            elif kernel_type == "polynomial":
                self.kernel_params = {"degree": 3, "coef0": 1.0, "gamma": 0.1}
            elif kernel_type == "laplacian":
                self.kernel_params = {"gamma": 0.1}
            elif kernel_type == "mixed":
                self.kernel_params = {"gamma_rbf": 0.1, "gamma_lap": 0.05, "degree": 2, "coef0": 1.0}
            else:
                raise ValueError(f"不支持的核类型: {kernel_type}")
        else:
            self.kernel_params = kernel_params
        
        # 初始化核中心点
        if self.learn_centers:
            self.centers = nn.Parameter(torch.randn(n_centers, input_size))
        else:
            self.register_buffer("centers", torch.randn(n_centers, input_size))
        
        # 从核特征到输出的线性变换
        self.output_layer = nn.Linear(n_centers, output_size)

        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化中心点以更好地覆盖输入空间
        nn.init.xavier_uniform_(self.centers)
        
        # 初始化输出层
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def _compute_kernel(self, x):
        """计算输入x与中心点之间的核值"""
        if self.kernel_type == "rbf":
            # RBF核: exp(-gamma * ||x-y||^2)
            gamma = self.kernel_params["gamma"]
            squared_diff = torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0))**2, dim=2)
            return torch.exp(-gamma * squared_diff)
        
        elif self.kernel_type == "polynomial":
            # 多项式核: (gamma * <x,y> + coef0)^degree
            gamma = self.kernel_params.get("gamma", 1.0)
            coef0 = self.kernel_params.get("coef0", 1.0)
            degree = self.kernel_params.get("degree", 3)
            dot_product = torch.matmul(x, self.centers.t())
            return (gamma * dot_product + coef0) ** degree
        
        elif self.kernel_type == "laplacian":
            # 拉普拉斯核: exp(-gamma * ||x-y||_1)
            gamma = self.kernel_params["gamma"]
            l1_diff = torch.sum(torch.abs(x.unsqueeze(1) - self.centers.unsqueeze(0)), dim=2)
            return torch.exp(-gamma * l1_diff)
        
        elif self.kernel_type == "mixed":
            # 混合核: 结合RBF和拉普拉斯核的优点
            gamma_rbf = self.kernel_params.get("gamma_rbf", 0.1)
            gamma_lap = self.kernel_params.get("gamma_lap", 0.05)
            degree = self.kernel_params.get("degree", 2)
            coef0 = self.kernel_params.get("coef0", 1.0)
            
            # RBF核部分
            squared_diff = torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0))**2, dim=2)
            rbf_kernel = torch.exp(-gamma_rbf * squared_diff)
            
            # 拉普拉斯核部分
            l1_diff = torch.sum(torch.abs(x.unsqueeze(1) - self.centers.unsqueeze(0)), dim=2)
            lap_kernel = torch.exp(-gamma_lap * l1_diff)
            
            # 组合两种核
            mixed_kernel = 0.7 * rbf_kernel + 0.3 * lap_kernel
            return mixed_kernel
    
    def forward(self, x):
        """
        通过基于核函数的扩展层的前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch_size, input_size)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            输出张量及门控值，形状为 (batch_size, output_size) 和 (output_size,)
        """
        # 计算核特征
        kernel_features = self._compute_kernel(x)  # (batch_size, n_centers)
        
        # 映射到输出空间
        outputs = self.output_layer(kernel_features)  # (batch_size, output_size)
            
        return outputs

class ResidualBlock(torch.nn.Module):
    """
    Residual block with SiLU activation and Layer Normalization
    """
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.block = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.SiLU(), 
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
        )
        
        # SiLU activation after the residual connection
        self.activation = torch.nn.SiLU()
        
        # Initialize with He/Kaiming initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.block.modules():
            if isinstance(m, torch.nn.Linear):
                # SiLU更适合使用kaiming_normal_初始化
                # 对于SiLU，可以使用与ReLU相似的初始化策略
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity  # Residual connection
        out = self.activation(out)
        return out

class Extensionlayer(torch.nn.Module):
    """
    Multi-timescale extension layer that captures relationships at different frequencies
    Implemented as a deep SiLU network with residual connections

    Parameters
    ----------
    input_size : int
        Dimension of the input features
    output_size : int
        Dimension of the output features
    hidden_size : int, default=128
        Size of hidden layers (128-1024 recommended)
    num_layers : int, default=10
        Number of hidden layers (10-20 recommended)
    dropout_rate : float, default=0.1
        Dropout rate for regularization
    """
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=10, dropout_rate=0.1):
        super(Extensionlayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection with SiLU
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.SiLU(inplace=True),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Stack of residual blocks
        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) for _ in range(num_layers)
        ])
        
        # Output projection with skip connection from input if possible
        self.output_proj = torch.nn.Linear(hidden_size, output_size)
        
        # Optional skip connection from input to output if dimensions allow
        self.has_direct_skip = (input_size <= output_size)
        if self.has_direct_skip:
            self.skip_proj = torch.nn.Linear(input_size, output_size, bias=False)
        
        # Initialize with He/Kaiming initialization
        self._init_weights()
    
    def _init_weights(self):
        # Input projection - 使用kaiming初始化适合SiLU
        torch.nn.init.kaiming_normal_(self.input_proj[0].weight, nonlinearity='relu')
        if self.input_proj[0].bias is not None:
            torch.nn.init.zeros_(self.input_proj[0].bias)
        
        # Output projection - 使用线性层的标准初始化
        torch.nn.init.kaiming_normal_(self.output_proj.weight, nonlinearity='linear')
        if self.output_proj.bias is not None:
            torch.nn.init.zeros_(self.output_proj.bias)
        
        # Skip connection (if exists)
        if self.has_direct_skip:
            torch.nn.init.xavier_uniform_(self.skip_proj.weight)
    
    def forward(self, x):
        """
        Forward pass through deep SiLU network with residual connections
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Store input for possible skip connection
        input_x = x
        
        # Input projection
        out = self.input_proj(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output projection
        result = self.output_proj(out)
        
        # Add skip connection if dimensions allow
        if self.has_direct_skip:
            skip = self.skip_proj(input_x)
            result = result + skip
        
        return result