import json
import math
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
ANGLE_SCALE_FACTOR = 10
DEFAULT_THINNING = 100
DEFAULT_DEVICE = "cpu"


class DataLoader:
    # TODO: Add train-test split here later
    def __init__(
        self,
        casp_version=None,
        thinning=DEFAULT_THINNING,
        device=DEFAULT_DEVICE,
    ) -> None:
        """Data Generator Class

        Args:
            file_path (str): File Path
            device (str): Device to Initialize tensors on
            train_split (float): Train vs Val Split
        """
        self.device = device
        # Data loading
        if casp_version:
            self.data = scn.load(
                casp_version=casp_version, thinning=thinning, scn_dataset=True
            )
        else:
            self.data = scn.load("debug", scn_dataset=True)

        # defining vocab
        with open("amino_acid_info.json", "r") as f:
            self.aa_data = json.load(f)

        unique_aas = [i["code"] for i in self.aa_data.values()]
        vocab = [PAD_TOKEN, MASK_TOKEN] + list(set(unique_aas))

        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        self.mask_seq_idx = self.char_to_ix[MASK_TOKEN]
        self.pad_seq_idx = self.char_to_ix[PAD_TOKEN]

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

    def fix_angles(self, angles: np.array) -> torch.tensor:
        """Change angular data from radians to degrees
           and converting neg angles to positive

        Args:
            angles (torch.tensor): Angular data in rads

        Returns:
            torch.tensor: Fixed angular data in degrees
        """
        angles_in_degrees = angles * (180 / math.pi)
        # Convert all neg angles to positive
        angles_in_degrees[:, :][angles_in_degrees[:, :] < 0] += 360
        angles_fixed = torch.tensor(
            angles_in_degrees * ANGLE_SCALE_FACTOR,
            device=self.device,
            dtype=torch.long,
        )
        return angles_fixed

    def generate_pairwise_distance(self, coords: np.array) -> torch.tensor:
        """Generates pairwise distance between residues.
        The distance is taken between c-alpha atoms of every residue

        Args:
            coords (np.array): all coordinates as an np array

        Returns:
            torch.tensor: pairwise distance
        """
        c_alpha_coords = coords[1::14]
        seq_len = c_alpha_coords.shape[0]
        pairwise_dist = torch.zeros(
            MAX_PROTEIN_SEQ_LEN, MAX_PROTEIN_SEQ_LEN, device=self.device
        )
        for i in range(seq_len):
            for j in range(seq_len):
                pairwise_dist[i][j] = np.linalg.norm(
                    c_alpha_coords[i] - c_alpha_coords[j]
                )
        return pairwise_dist

    def mask_and_pad_sequences(
        self,
        batch_data: dict,
        masking_percent=0.15,
        mask_angles=True,
    ) -> dict:
        """Mask and Pad Sequences

        Args:
            batch_data (dict): Batch data including seq_data, angles
            masking_percent (float, optional): _description_. Defaults to 0.15.
            mask_angles (bool, optional): _description_. Defaults to True.

        Returns:
            dict: Masked & Padded seq and angular data
        """
        batch_size = len(batch_data["seq_data"])
        batch_data["masked_idxs"] = []
        batch_data["seq_masked"] = torch.full(
            (batch_size, MAX_PROTEIN_SEQ_LEN),
            self.pad_seq_idx,
            device=self.device,
        )
        batch_data["angles_masked"] = torch.full(
            (batch_size, MAX_PROTEIN_SEQ_LEN, 6),
            self.pad_angle_idx,
            device=self.device,
        )

        for seq_idx in range(batch_size):
            seq = batch_data["seq_data"][seq_idx].detach().clone()
            angles = batch_data["angle_data"][seq_idx].detach().clone()

            # how many masks per sequence
            n_residue = len(seq)
            n_masks = int(masking_percent * n_residue)
            masked_idxs = torch.randint(n_residue, (n_masks,))

            for mask_idx in masked_idxs:
                seq[mask_idx] = self.mask_seq_idx

                if mask_angles:
                    angles[mask_idx] = self.mask_angle_idx

            batch_data["masked_idxs"].append(masked_idxs)
            batch_data["seq_masked"][seq_idx, : len(seq)] = seq
            batch_data["angles_masked"][seq_idx, : len(angles)] = angles

            # Pad the original seq and angles data
            to_pad = MAX_PROTEIN_SEQ_LEN - len(seq)
            batch_data["seq_data"][seq_idx] = nn.functional.pad(
                batch_data["seq_data"][seq_idx],
                [0, to_pad],
                "constant",
                self.pad_seq_idx,
            )
            batch_data["angle_data"][seq_idx] = nn.functional.pad(
                batch_data["angle_data"][seq_idx],
                [0, 0, 0, to_pad],
                "constant",
                self.pad_angle_idx,
            )

        # Concat original data list into a single tensor
        for key in ["seq_data", "angle_data", "pairwise_dist"]:
            batch_data[key] = torch.stack(batch_data[key])

        return batch_data

    def generate_batch(
        self,
        batch_size,
        masking_percent=DEFAULT_MASKING_PERCENT,
    ) -> GeneratorExit:

        print("Starting Generator")
        total_batches = int(len(self.data) / batch_size)

        for curr_batch in range(total_batches):
            print(f"Starting Batch: {curr_batch}")
            batch_items = self.data[curr_batch : curr_batch + batch_size]

            # Store all info in the batch_data dict
            batch_data = {
                "seq_data": [],
                "angle_data": [],
                "pairwise_dist": [],
            }

            for seq in batch_items:
                batch_data["seq_data"].append(
                    torch.tensor(
                        [self.char_to_ix[i] for i in seq.seq],
                        device=self.device,
                        dtype=torch.long,
                    )
                )
                # take the backbone torsion and bond angles (first 6 angles)
                batch_data["angle_data"].append(
                    self.fix_angles(seq.angles[:, :6])
                )
                # take only the c-alpha carbon co-ordinates
                batch_data["pairwise_dist"].append(
                    self.generate_pairwise_distance(seq.coords)
                )

            # yield batch_data
            # mask masking_percent of original compound sequences
            print("Masking...")
            batch_data_processed = self.mask_and_pad_sequences(
                batch_data, masking_percent=masking_percent
            )
            yield batch_data_processed


class Cermit(GPT):
    def __init__(
        self,
        num_blocks: int,
        n_heads: int,
        emb_size: int,
        data_generator: DataLoader,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:

        super().__init__(
            num_blocks=num_blocks,
            n_heads=n_heads,
            emb_size=emb_size,
            block_size=MAX_PROTEIN_SEQ_LEN,
            data_generator=data_generator,
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
