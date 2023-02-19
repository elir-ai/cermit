import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import sidechainnet as scn
from nano_gpt import GPT

MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_MASKING_PERCENT = 0.15
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_ACTIVATION = "relu"
MAX_PROTEIN_SEQ_LEN = 2600
MAX_PROTEIN_ANGLE = 1800
MAX_PROTEIN_COORD = 10000
TOTAL_STRUCTURAL_VARS = 54
TOTAL_ANGLES = 12
TOTAL_COORDINATES = 42
STRUCTURE_SCALING_FACTOR = 10


class DataGenerator:
    # TODO: Add train-test split here later
    def __init__(self, device: str) -> None:
        """Data Generator Class

        Args:
            file_path (str): File Path
            device (str): Device to Initialize tensors on
            train_split (float): Train vs Val Split
        """
        self.device = device
        # Data loading
        self.data = scn.load(casp_version=9, thinning=100, scn_dataset=True)

        # defining vocab
        with open("amino_acid_info.json", "r") as f:
            self.aa_data = json.load(f)

        unique_aas = [i["code"] for i in self.aa_data.values()]
        vocab = [PAD_TOKEN, MASK_TOKEN] + list(set(unique_aas))

        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        self.mask_token_idx = self.char_to_ix[MASK_TOKEN]
        self.pad_token_idx = self.char_to_ix[PAD_TOKEN]

        self.mask_angle_idx = MAX_PROTEIN_ANGLE
        self.pad_angle_idx = MAX_PROTEIN_ANGLE + 1

        self.mask_coords_idx = MAX_PROTEIN_COORD
        self.pad_coords_idx = MAX_PROTEIN_COORD + 1

        # encode data into train & eval batches
        # self.aa_data, self.struct_data = self.encode(self.data)
        # self.data_encoded = [self.encode(smi_str) for smi_str in self.data]

    def encode(self, smi_str: str) -> list:
        return torch.tensor([self.char_to_ix[char] for char in smi_str])

    def decode(self, list_ix):
        smi_string = ""
        for ix in list_ix:
            smi_string += self.ix_to_char[ix]
        return smi_string

    def mask_sequences(
        self,
        aa_data,
        struct_data,
        masking_percent=0.15,
        mask_struct=True,
    ):
        all_masked_idxs = []
        all_aa_masked = []
        all_struct_masked = []

        for seq_idx in range(len(aa_data)):
            aa_masked = aa_data[seq_idx].detach().clone()
            struct_masked = struct_data[seq_idx].detach().clone()

            # how many masks per sequence
            n_residue = len(aa_masked)
            n_masks = int(masking_percent * n_residue)
            masked_idxs = torch.randint(n_residue, (n_masks,))

            for mask_idx in masked_idxs:
                aa_masked[mask_idx] = self.mask_token_idx

                if mask_struct:
                    struct_masked[mask_idx][
                        :TOTAL_COORDINATES
                    ] = self.mask_coords_idx
                    struct_masked[mask_idx][
                        TOTAL_COORDINATES:
                    ] = self.mask_angle_idx

            all_masked_idxs.append(masked_idxs)
            all_aa_masked.append(aa_masked)
            all_struct_masked.append(struct_masked)

        return all_aa_masked, all_struct_masked, all_masked_idxs

    def generate_batch(
        self,
        batch_size,
        masking_percent=DEFAULT_MASKING_PERCENT,
    ):

        print("Started")
        # generate random indexes
        ixes = torch.randint(
            len(self.data),
            (batch_size,),
            device=self.device,
        )

        batch = [self.data[ix.item()] for ix in ixes]
        print("Generated random batch with index")
        # generate amino acid sequence with its corresponding
        # structural information
        aa_data = []
        struct_data = []
        for seq in batch:
            aa_data.append(
                torch.tensor(
                    [self.char_to_ix[i] for i in seq.seq],
                    device=self.device,
                    dtype=torch.long,
                )
            )
            # flatten out the xyz for all 14 co-ordinates to a TOTAL_COORDINATES len feat vector per residue
            coords = seq.coords.reshape(-1, 14 * 3)
            angles = seq.angles
            struct_data.append(
                torch.tensor(
                    np.concatenate([coords, angles], axis=1)
                    * STRUCTURE_SCALING_FACTOR,
                    device=self.device,
                    dtype=torch.long,
                )
            )
        print("Flattened all struct_info")

        # return aa_data, struct_data
        # mask masking_percent of original compound sequences
        aa_masked, struct_masked, masked_idxs = self.mask_sequences(
            aa_data[:], struct_data[:], masking_percent=masking_percent
        )
        print("Masked")
        # return aa_data, struct_data, aa_masked, struct_masked, masked_idxs

        # pad sequences
        aa_original_padded = torch.empty(
            batch_size,
            MAX_PROTEIN_SEQ_LEN,
            device=self.device,
            dtype=torch.long,
        )
        aa_masked_padded = torch.empty(
            batch_size,
            MAX_PROTEIN_SEQ_LEN,
            device=self.device,
            dtype=torch.long,
        )
        struct_masked_padded = torch.empty(
            batch_size,
            MAX_PROTEIN_SEQ_LEN,
            54,
            device=self.device,
            dtype=torch.long,
        )
        padded_mask = torch.zeros(
            batch_size,
            MAX_PROTEIN_SEQ_LEN,
            device=self.device,
            dtype=torch.bool,
        )

        for batch_idx in range(batch_size):
            right_pad = MAX_PROTEIN_SEQ_LEN - len(aa_data[batch_idx])

            # pad original aa sequence
            aa_original_padded[batch_idx] = nn.functional.pad(
                aa_data[batch_idx],
                [0, right_pad],
                "constant",
                self.pad_token_idx,
            )

            # pad masked aa sequence
            aa_masked_padded[batch_idx] = nn.functional.pad(
                aa_masked[batch_idx],
                [0, right_pad],
                "constant",
                self.pad_token_idx,
            )

            # pad masked coords sequence
            coord_pad = nn.functional.pad(
                struct_masked[batch_idx][:, :TOTAL_COORDINATES],
                [0, 0, 0, right_pad],
                "constant",
                self.pad_coords_idx,
            )

            # pad masked angle sequence
            angle_pad = nn.functional.pad(
                struct_masked[batch_idx][:, TOTAL_COORDINATES:],
                [0, 0, 0, right_pad],
                "constant",
                self.pad_angle_idx,
            )

            # concat both and update struct_masked_padded
            struct_masked_padded[batch_idx] = torch.concat(
                [coord_pad, angle_pad], dim=-1
            )

            padded_mask[batch_idx, -right_pad:] = True
        print("padded!")
        return (
            aa_original_padded,
            aa_masked_padded,
            struct_masked_padded,
            masked_idxs,
            padded_mask,
        )


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
        self.gpt = GPT(
            num_blocks=num_blocks,
            n_heads=n_heads,
            emb_size=emb_size,
            block_size=MAX_PROTEIN_SEQ_LEN,
            data_generator=self.data_generator,
            device=self.device,
            dropout=dropout,
        )

    def forward(
        self,
        aa_masked: torch.tensor,
        struct_masked: torch.tensor,
        src_key_padding_mask: torch.tensor,
        aa_original=None,
        masked_idxs=None,
    ) -> torch.tensor:
        """Forward function for cermit

        Args:
            #TODO: Update args
            masked_X (torch.tensor): Masked indices of compound in batches.
                Size: batch_size * MAX_MOLECULE_LENGTH

        Returns:
            torch.tensor: _description_
        """
        logits = self.gpt(masked, src_key_padding_mask)
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

    def train_model(
        self,
        batch_size: int,
        num_epochs: int,
        checkpoint_itvl=10,
        lr=3e-4,
        save_model=False,
    ):
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            (
                original,
                masked,
                masked_idxs,
                padded_mask,
            ) = self.data_generator.generate_batch("train", batch_size)

            # get logits and loss
            _, loss = self(masked, padded_mask, original, masked_idxs)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            print(f"Epoch: {epoch}; Loss: {loss.item()}")
            opt.step()
