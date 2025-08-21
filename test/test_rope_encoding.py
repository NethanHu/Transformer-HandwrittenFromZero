import math

import numpy as np
import torch

from RoPE import rope_attention_score, rope_encoding


def demonstrate_relative_property():
    """
    演示RoPE的相对位置特性
    """
    d_model = 8
    q = torch.randn(d_model)
    k = torch.randn(d_model)

    print(f"Original Q vector: {q}")
    print(f"Original K vector: {k}")
    print(f"Original dot product: {torch.dot(q, k):.4f}\n")

    positions_distance_3 = [(0, 3), (1, 4), (2, 5), (5, 8), (10, 13)]
    scores_distance_3 = []
    for q_pos, k_pos in positions_distance_3:
        score = rope_attention_score(q, k, q_pos, k_pos)
        scores_distance_3.append(score.item())
        print(f"Position ({q_pos}, {k_pos}): attention score = {score:.4f}")

def compare_with_absolute_encoding():
    """
    与绝对位置编码对比
    """
    print(f"\n=== Compare with sinusoidal position embedding ===")

    d_model = 8
    q = torch.randn(d_model)
    k = torch.randn(d_model)

    # 简单的绝对位置编码（正弦/余弦）
    def absolute_pos_encoding(pos, d_model):
        encoding = torch.zeros(d_model)
        for i in range(0, d_model, 2):
            encoding[i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                encoding[i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        return encoding

    abs_scores = []
    rope_scores = []

    positions = [(0, 1), (1, 2), (10, 11), (98, 99)]  # 相对距离都是1

    for q_pos, k_pos in positions:
        q_abs = q + absolute_pos_encoding(q_pos, d_model)
        k_abs = k + absolute_pos_encoding(k_pos, d_model)
        abs_score = torch.dot(q_abs, k_abs)
        abs_scores.append(abs_score.item())

        rope_score = rope_attention_score(q, k, q_pos, k_pos)
        rope_scores.append(rope_score.item())

        print(f"Position ({q_pos},{k_pos}): sinusoidal ={abs_score:.4f}, RoPE = {rope_score:.4f}")

    print(f"\nsinusoidal score std: {np.std(abs_scores):.6f}")
    print(f"RoPE score std: {np.std(rope_scores):.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    demonstrate_relative_property()
    compare_with_absolute_encoding()
    print("RoPE Position Embedding test passed!")