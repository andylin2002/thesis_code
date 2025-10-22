import torch

# -----------------------------
# 建立 Hankel 矩陣
# -----------------------------
def construct_hankel_matrix(vector: torch.Tensor, row: int, col: int) -> torch.Tensor:

        # 'unfold' function needs the unsqueezing dimension
        vector_unfoldable = vector.unsqueeze(1) # (QT, 1, M)
        hankel_unfolded = vector_unfoldable.unfold(2, col, 1) # (QT, 1, row, col)
        hankel_matrix = hankel_unfolded.squeeze(1).contiguous() # (QT, row, col)

        return hankel_matrix

# -----------------------------
# 組成 enhance matrix
# -----------------------------
def construct_enhance_matrix(matrix: torch.Tensor) -> torch.Tensor:
    alpha = 5
    beta = 6
    
    C1 = construct_hankel_matrix(matrix[:, 0, :], alpha, beta)
    C2 = construct_hankel_matrix(matrix[:, 1, :], alpha, beta)
    C3 = construct_hankel_matrix(matrix[:, 2, :], alpha, beta)

    row1 = torch.cat([C1, C2], dim=2)  # (B, 5, 12)
    row2 = torch.cat([C2, C3], dim=2)  # (B, 5, 12)
    enhance_matrix = torch.cat([row1, row2], dim=1)  # (B, 10, 12)

    return enhance_matrix

# -----------------------------
# 建立輸入 tensor
# -----------------------------
matrix = torch.tensor([
    [
        [1,2,3,4,5,6,7,8,9,10],
        [11,12,13,14,15,16,17,18,19,20],
        [21,22,23,24,25,26,27,28,29,30]
    ]
], dtype=torch.float32)

enhance_matrix = construct_enhance_matrix(matrix)

print("Input shape:", matrix.shape)
print("Enhance matrix shape:", enhance_matrix.shape)
print("Enhance matrix:\n", enhance_matrix)
