# Karpathy = Chad
import os
from abc import ABCMeta
from abc import abstractmethod

import torch
from torch import nn

DEFAULT_DROPOUT_RATE = 0.4
MODEL_NAME = "nanoGPT"


class FeedFowardLayer(nn.Module):
    def __init__(
        self, emb_size: int, dropout=DEFAULT_DROPOUT_RATE, scaling_factor=4
    ) -> None:
        """FeedForward Layer that comes after Multi-Head Attention.
            Scales the output of MHA from head_size to emb_size.

        Args:
            emb_size (int): Embedding Size
            dropout (float, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
            scaling_factor (int, optional): Scaling Factor among the two Linear Layers.
                Defaults to 4.
        """
        super().__init__()
        self.emb_size = emb_size
        self.feed_foward = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size * scaling_factor),
            nn.GeLU(),
            nn.Linear(self.emb_size * scaling_factor, self.emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            X (torch.tensor): Should be the output of MHA. Output shape: B, T, head_size

        Returns:
            torch.tensor: Output shape: B, T, emb_size
        """
        return self.feed_foward(X)  # B, T, emb_size


class ScaledAttention(nn.Module):
    def __init__(
        self,
        head_size: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """ScaledAttention Block.
            Generates K, Q, V vectors of size: head_size to calculate weighted attention (Q.K)
                and returns (weighted_attention).V

        Args:
            head_size (int): Single Head Size.
                Generally MHA head_size / n_heads
            emb_size (int): Embedding Size.
            block_size (int): Block Size.
            dropout (int, DEFAULT_DROPOUT_RATE): Dropout Rate.
                Defaults to DEFAULT_DROPOUT_RATE
        """
        super().__init__()
        self.head_size = head_size
        self.l_key = nn.Linear(emb_size, head_size, bias=False)
        self.l_query = nn.Linear(emb_size, head_size, bias=False)
        self.l_value = nn.Linear(emb_size, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(block_size, block_size, dtype=torch.float32)
            ),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, X: torch.tensor, padding_mask: torch.tensor
    ) -> torch.tensor:
        """Forward Function

        Args:
            X (torch.tensor): X should be the output of sem_emb + pos_emb of shape B, T, emb_size

        Returns:
            torch.tensor: Output of shape B, T, head_size
        """
        B, T, C = X.shape
        Q = self.l_query(X)  # B, T, head_size
        K = self.l_key(X)  # B, T, head_size
        V = self.l_value(X)  # B, T, head_size
        # Produce weights
        wei = Q @ K.transpose(-2, -1) * C**-0.5  # B, T, T
        # copy padded from B, T -> B, T, T
        padding_mask_ = padding_mask.view(B, 1, T).expand(B, T, T)
        wei = wei.masked_fill(padding_mask_, float("-inf"))
        wei = wei.softmax(-1)  # B, T, T
        wei = self.dropout(wei)
        out = wei @ V  # B, T, head_size

        # TODO: mask out of padded tokens to 0
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """Multi-Headed Attention Class

        Args:
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.n_heads = n_heads
        self.attention_blocks = nn.ModuleList(
            [
                ScaledAttention(
                    head_size=emb_size // self.n_heads,
                    emb_size=emb_size,
                    block_size=block_size,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj_layer = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, X: torch.tensor, padding_mask: torch.tensor
    ) -> torch.tensor:
        """Forward Function for Multi-Headed Self-Attention

        Args:
            X (torch.tensor): Input to MHA block. Of Shape B, T, emb_size

        Returns:
            torch.tensor: _description_
        """
        # print(f"X.shape: {X.shape}, padding_mask.shape: {padding_mask.shape}")
        out = torch.cat(
            [block(X, padding_mask) for block in self.attention_blocks], -1
        )
        # B, T, head_size -> B, T, emb_size
        out = self.dropout(self.proj_layer(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """Attention Block consisting of MHA, FeedForward, Residual connections, and LayerNorm

        Args:
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.mha = MultiHeadedAttention(
            n_heads=n_heads,
            emb_size=emb_size,
            block_size=block_size,
            dropout=dropout,
        )
        self.ff = FeedFowardLayer(emb_size=emb_size, dropout=dropout)
        # Layer Norm Layers
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, comb_input: torch.tensor) -> torch.tensor:
        """Forward Function for Attention Block

        Args:
            comb_input (torch.tensor): Should be a tuple of (embedding, padded mask).
                - embedding is the output of (sem_emb + pos_emb) of shape: B, T, emb_size
                - padding_mask (the same size) should denote positions of padded tokens
        Returns:
            torch.tensor: _description_
        """
        # because sequential only takes one input
        X, padding_mask = comb_input
        X = self.mha(self.ln1(X), padding_mask) + X
        X = self.ff(self.ln2(X)) + X
        return (X, padding_mask)


class GPT(nn.Module, metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):
        if cls is GPT:
            raise TypeError(f"Can't instantiate abstract class {cls.__name__}")
        return object.__new__(cls)

    @abstractmethod
    def __init__(
        self,
        num_blocks: int,
        n_heads: int,
        emb_size: int,
        block_size: int,
        vocab_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """GPT Module

        Args:
            num_blocks (int): Num blocks of the attention block.
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            data_generator (DataGenerator): Data Generator
            device (str): Device to store params GPT params on.
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.block_size = block_size
        self.semantic_embedding_table = nn.Embedding(
            vocab_size, emb_size, dtype=torch.float32
        )
        self.positional_emb_table = nn.Embedding(
            self.block_size, emb_size, dtype=torch.float32
        )
        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(
                    n_heads=n_heads,
                    emb_size=emb_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_layer = nn.Linear(emb_size, vocab_size)
        self.ln_f = nn.LayerNorm(emb_size)

    def forward(
        self,
        aa_padded: torch.tensor,
        struct_padded: torch.tensor,
        padding_mask: torch.tensor,
    ) -> torch.tensor:
        sem_emb = self.semantic_embedding_table(aa_padded)  # B, T, emb_size
        # TODO: Check if position start from 0 or 1
        pos_emb = self.positional_emb_table(
            torch.arange(aa_padded.shape[1], device=self.device)
        )  # T, emb_size
        coord_emb = self.coord_emb_table(struct_padded[:, :, :42])
        ang_emb = self.angle_emb_table(struct_padded[:, :, 42:])
        comb_emb = sem_emb + pos_emb + coord_emb + ang_emb
        # _ => padding_mask returned by sequential
        out, _ = self.attention_layers((comb_emb, padding_mask))
        out = self.ln_f(out)  # B, T, emb_size
        return self.linear_layer(out)  # B, T, vocab_size
