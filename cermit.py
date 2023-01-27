import pandas as pd
import torch
from torch import nn

MASK_TOKEN = "[MASK]"
PAD_TOKEN = "PAD"
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_MASKING_PERCENT = 0.15
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_ACTIVATION = "relu"
MAX_MOLECULE_SIZE = 120


class DataGenerator:
    def __init__(self, file_path: str, device: str) -> None:
        """Data Generator Class

        Args:
            file_path (str): File Path
            device (str): Device to Initialize tensors on
            train_split (float): Train vs Val Split
        """
        self.device = device
        # Data loading
        with open(file_path, "r") as f:
            self.data = pd.read_csv(
                file_path, names=["unk", "smiles"], header=None
            )["smiles"].values

        # defining vocab
        unique_chars = []
        for smile in self.data:
            unique_chars += list(str(set(smile)))

        vocab = [PAD_TOKEN, MASK_TOKEN] + list(set(unique_chars))
        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        self.mask_token_idx = self.char_to_ix[MASK_TOKEN]
        self.pad_token_idx = self.char_to_ix[PAD_TOKEN]

        # encode data into train & eval batches
        self.data_encoded = [self.encode(smi_str) for smi_str in self.data]

    def encode(self, smi_str: str) -> list:
        return torch.tensor([self.char_to_ix[char] for char in smi_str])

    def decode(self, list_ix):
        smi_string = ""
        for ix in list_ix:
            smi_string += self.ix_to_char[ix]
        return smi_string

    def mask_sequences(self, sequences, masking_percent=0.15):
        masked_idxs = []
        masked = []
        for seq in sequences:
            masked_seq = seq.clone().detach()
            n_masks = int(masking_percent * len(seq))
            masked_idx = torch.randint(len(seq), (n_masks,))

            for idx in masked_idx:
                masked_seq[idx] = self.mask_token_idx

            masked.append(masked_seq)
            masked_idxs.append(masked_idx)
        return sequences, masked, masked_idxs

    def generate_batch(
        self,
        split_type,
        batch_size,
        train_split=DEFAULT_TRAIN_SPLIT,
        masking_percent=DEFAULT_MASKING_PERCENT,
    ):
        split_idx = int(train_split * len(self.data_encoded))
        if split_type == "train":
            batch_to_generate_from = self.data_encoded[:split_idx]
        else:
            batch_to_generate_from = self.data_encoded[split_idx:]

        # generate random indexes
        ixes = torch.randint(
            len(batch_to_generate_from),
            (batch_size,),
            device=self.device,
        )

        original = [batch_to_generate_from[ix] for ix in ixes]

        # mask masking_percent of original compound sequences
        original, masked, masked_idxs = self.mask_sequences(
            original, masking_percent=masking_percent
        )

        original_padded = torch.empty(
            batch_size, MAX_MOLECULE_SIZE, device=self.device, dtype=torch.long
        )
        masked_padded = torch.empty(
            batch_size, MAX_MOLECULE_SIZE, device=self.device, dtype=torch.long
        )
        padded_mask = torch.empty(
            batch_size, MAX_MOLECULE_SIZE, device=self.device, dtype=torch.bool
        )

        for idx in range(batch_size):
            right_pad = MAX_MOLECULE_SIZE - len(original[idx])

            # pad original sequence
            original_padded[idx] = nn.functional.pad(
                original[idx],
                [0, right_pad],
                "constant",
                self.pad_token_idx,
            )

            # pad masked sequence
            masked_padded[idx] = nn.functional.pad(
                masked[idx],
                [0, right_pad],
                "constant",
                self.pad_token_idx,
            )
            padded_mask[idx][-right_pad:] = True

        return original_padded, masked_padded, masked_idxs, padded_mask


class Cermit(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        n_heads: int,
        emb_size: int,
        data_generator: DataGenerator,
        device: str,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:

        super().__init__()
        self.device = device
        self.data_generator = data_generator
        self.semantic_embedding_table = nn.Embedding(
            self.data_generator.vocab_size, emb_size
        )
        self.positional_emb_table = nn.Embedding(MAX_MOLECULE_SIZE, emb_size)

        # define a transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation=DEFAULT_ACTIVATION,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks
        )

        # Do a final Layer Norm
        self.ln_f = nn.LayerNorm(emb_size)

        # Pass into the linear layer to get probabilities
        self.linear_layer = nn.Linear(emb_size, self.data_generator.vocab_size)

    def forward(
        self,
        masked: torch.tensor,
        src_key_padding_mask: torch.tensor,
        original: None,
        masked_idxs: None,
    ) -> torch.tensor:
        """Forward function for cermit

        Args:
            #TODO: Update args
            masked_X (torch.tensor): Masked indices of compound in batches.
                Size: batch_size * MAX_MOLECULE_LENGTH

        Returns:
            torch.tensor: _description_
        """
        sem_emb = self.semantic_embedding_table(masked)
        pos_emb = self.positional_emb_table(torch.arange(0, MAX_MOLECULE_SIZE))
        out = self.transformer_encoder(
            src=sem_emb + pos_emb, src_key_padding_mask=src_key_padding_mask
        )
        logits = self.linear_layer(self.ln_f(out))
        B, T, V = logits.shape

        loss = torch.zeros(1, device=self.device)

        # calculate loss
        if original is not None and masked_idxs is not None:
            loss_func = nn.CrossEntropyLoss()
            for batch_idx in range(B):
                # get masked index of tokens for the particular batch
                masked_idx = masked_idxs[batch_idx]

                # output for that batch_index
                loss += loss_func(
                    logits[batch_idx, masked_idx],
                    original[batch_idx, masked_idx],
                )
            loss /= B

        return logits, loss
