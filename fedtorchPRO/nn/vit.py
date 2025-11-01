import torch
import torch.nn as nn
import torch.nn.functional as F

from fedtorchPRO.imitate.transforms import Resize

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, N, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)

        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, E)  # (B, N, embed_dim)

        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.layer_norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.layer_norm2(x + self.dropout2(ffn_out))

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, num_heads=12, num_layers=12, hidden_dim=3072, dropout=0.):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
        self.resizef = Resize((224,224))

    def forward(self, x):
        x = self.patch_embedding(self.resizef(x))  # (B, num_patches, embed_dim)
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((class_token, x), dim=1)  # (B, num_patches + 1, embed_dim)

        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 0]  # (B, embed_dim) - 取出 class token
        return self.fc_out(x)  # (B, num_classes)

def vitsmall(num_classes = 10):
    return VisionTransformer(num_classes=num_classes,embed_dim=256,num_heads=8,num_layers=6)

# 示例使用
if __name__ == "__main__":
    model = VisionTransformer()
    x = torch.randn(8, 3, 224, 224)  # (batch_size, channels, height, width)
    logits = model(x)
    print(logits.shape)  # 应该输出 (8, 1000)
