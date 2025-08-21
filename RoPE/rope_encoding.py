import torch


def rope_encoding(x: torch.Tensor, position: int, base: float = 10000.0) -> torch.Tensor:
    """
    RoPE旋转位置编码
    一个二维向量 (x, y) 可以被看作一个复数 x + iy。欧拉公式告诉我们：e^(iθ) = cos(θ) + i*sin(θ)。
    在复数平面上，将一个复数 x + iy 旋转 θ 角度，等价于乘以 e^(iθ)。
    乘法：(x + iy) * (cosθ + isinθ) = (x cosθ - y sinθ) + i(x sinθ + y cosθ)
    结果：旋转后的新向量就是 (x cosθ - y sinθ, x sinθ + y cosθ)

    Args:
        x: 输入向量 [d_model]
        position: token 所在的位置索引 (0, 1, 2, ...)
        base: 频率基数，默认10000

    Returns:
        旋转后的向量 [d_model]
    """
    d_model: int = x.shape[-1]
    assert d_model % 2 == 0, "d_model must be even for RoPE"
    # 旋转编码不再需要对每个位置计算一个编码，而是创新地将 d_model 中两两成对做 2D 旋转
    i: torch.Tensor = torch.arange(0, d_model // 2, dtype=torch.float32)
    # 每个组对应的角速度（更严格说是“逆频率”）
    theta: torch.Tensor = base ** (-2 * i / d_model)
    # 低维组 i 小，角速度大（变化更快）；高维组 i 大，角速度小（变化更慢），这与传统正弦位置编码一样是多尺度频率
    angles: torch.Tensor = position * theta
    cos_angles: torch.Tensor = torch.cos(angles)
    sin_angles: torch.Tensor = torch.sin(angles)
    # 把 [d] 切成 [d/2, 2] 的小二维向量；分别当作复数的实部/虚部
    x_pairs: torch.Tensor = x.view(-1, 2)
    x_real: torch.Tensor = x_pairs[:, 0]
    x_imag: torch.Tensor = x_pairs[:, 1]
    # (x cosθ - y sinθ, x sinθ + y cosθ)
    rotated_real: torch.Tensor = x_real * cos_angles - x_imag * sin_angles
    rotated_imag: torch.Tensor = x_real * sin_angles + x_imag * cos_angles
    # 重新组合成原来的尺寸
    rotated: torch.Tensor = torch.stack([rotated_real, rotated_imag], dim=-1)  # [d_model/2, 2]
    rotated: torch.Tensor = rotated.view(-1)  # [d_model]

    return rotated


def rope_attention_score(q: torch.Tensor, k: torch.Tensor, q_pos: int, k_pos: int) -> torch.Tensor:
    q_rope: torch.Tensor = rope_encoding(q, q_pos)
    k_rope: torch.Tensor = rope_encoding(k, k_pos)
    score: torch.Tensor = torch.dot(q_rope, k_rope)
    return score