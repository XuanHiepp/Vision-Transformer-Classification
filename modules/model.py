from torch import nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()

        self.patch_split = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                     kernel_size=patch_size, stride=patch_size, padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        # only flatten the feature map dimensions into a single vector
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x_patches = self.patch_split(x)
        x_flatten = self.flatten(x_patches)

        return x_flatten.permute(0, 2, 1)


# Create Transformer Block
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim)  # Layer norm
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False)
        return attn_output


# Create MLP Block
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, hidden_size=3072, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim)  # Layer norm
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_size),
            nn.GELU(),  # GELU activation
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# Create Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()

        self.mha_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  hidden_size=mlp_size,
                                  dropout=mlp_dropout)

    # 5. Create a forward() method
    def forward(self, x):
        x = self.mha_block(x) + x
        x = self.mlp_block(x) + x
        return x


# Creates Vision Transformer Block
class ViTBlock(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12, embedding_dim=768, mlp_size=3072, num_heads=12,
                 attn_dropout=0, mlp_dropout=0.1, embedding_dropout=0.1, num_classes=1000):
        super().__init__()

        # number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create class embedding
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True)

        # Create position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(
            1, self.num_patches+1, embedding_dim), requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # Create Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # Create classifier block
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]  # batch size

        class_token = self.class_embedding.expand(
            batch_size, -1, -1)  # Class token embedding
        x = self.patch_embedding(x)  # Patch embedding
        # Concat class and patch embedding
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x  # Add position embedding to patch embedding

        x = self.embedding_dropout(x)  # embedding dropout
        x = self.transformer_encoder(x)  # transformer encoder layers
        # classifier layer, run on each sample in a batch at 0 index
        x = self.classifier(x[:, 0])

        return x
