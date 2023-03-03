import os
import shutil
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
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
MAX_PROTEIN_SEQ_LEN = 1000
MAX_PROTEIN_ANGLE = 3600
MAX_PROTEIN_COORD = 10000
TOTAL_STRUCTURAL_VARS = 54
TOTAL_ANGLES = 12
TOTAL_COORDINATES = 42
ANGLE_SCALE_FACTOR = 10
DEFAULT_THINNING = 100
DEFAULT_DEVICE = "cpu"
BASE_DIR = os.path.dirname(__file__)

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
        scn_dir = BASE_DIR + "/sidechainnet_data"
        if casp_version:
            self.data = scn.load(
                casp_version=casp_version, thinning=thinning, scn_dataset=True, scn_dir=scn_dir
            )
        else:
            self.data = scn.load("debug", scn_dataset=True, scn_dir=scn_dir)

        # defining vocab
        with open(f"{BASE_DIR}/amino_acid_info.json", "r") as f:
            self.aa_data = json.load(f)

        unique_aas = [i["code"] for i in self.aa_data.values()]
        vocab = [PAD_TOKEN, MASK_TOKEN] + list(set(unique_aas))

        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        self.mask_seq_idx = self.char_to_ix[MASK_TOKEN]
        self.pad_seq_idx = self.char_to_ix[PAD_TOKEN]

        self.pad_angle_idx = MAX_PROTEIN_ANGLE
        self.mask_angle_idx = MAX_PROTEIN_ANGLE + 1

    def trim_data(self) -> None:
        self.data = [prot for prot in self.data if len(prot) <= MAX_PROTEIN_SEQ_LEN]

    def encode(self, smi_str: str) -> list:
        return torch.tensor([self.char_to_ix[char] for char in smi_str], dtype=torch.float32)

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
            dtype=torch.int,
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
        c_alpha_coords = torch.tensor(coords[1::14], device=self.device, dtype=torch.float16)
        diff = c_alpha_coords[:, None, :] - c_alpha_coords[None, :, :]
        pairwise_dist = torch.sqrt(torch.sum(diff**2, axis=-1))
        pairwise_dist_padded = torch.empty(
            MAX_PROTEIN_SEQ_LEN, MAX_PROTEIN_SEQ_LEN, device=self.device, dtype=torch.float32
        )
        seq_len = len(pairwise_dist)
        pairwise_dist_padded[:seq_len, :seq_len] = pairwise_dist
        return pairwise_dist_padded

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
        batch_data["padding_mask"] = torch.zeros(
            batch_size,
            MAX_PROTEIN_SEQ_LEN,
            device=self.device,
            dtype=torch.bool,
        )

        for seq_idx in range(batch_size):
            seq = batch_data["seq_data"][seq_idx].detach().clone()
            angles = batch_data["angle_data"][seq_idx].detach().clone()

            # how many masks per sequence
            n_residue = len(seq)
            n_masks = int(masking_percent * n_residue)
            # randperm is used to ensure unique indexes for masks are produced :)
            masked_idxs = torch.randperm(n_residue, device=self.device)[:n_masks]

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
            batch_data["padding_mask"][seq_idx, -to_pad:] = True

        # Stack original data list into a single tensor
        for key in ["seq_data", "angle_data", "pairwise_dist"]:
            batch_data[key] = torch.stack(batch_data[key])

        return batch_data

    def generate_batch(
        self,
        batch_size,
        masking_percent=DEFAULT_MASKING_PERCENT,
    ) -> GeneratorExit:

        self.trim_data()

        total_batches = len(self.data) // batch_size
        for curr_batch in range(total_batches):
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
                        dtype=torch.int,
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

            # mask masking_percent of original compound sequences
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
        data_loader: DataLoader,
        dropout=DEFAULT_DROPOUT_RATE,
        model_name="cermit",
    ) -> None:
        """Cermit Class inherits base GPT class

        Args:
            num_blocks (int): Num Attention blocks
            n_heads (int): Num attention heads in MHA
            emb_size (int): Emb Size of every token. Head size = Emb size // n_heads
            data_loader (DataLoader): Data Loader class
            dropout (float, optional): Dropout. Defaults to DEFAULT_DROPOUT_RATE.
        """
        self.model_name = model_name
        self.data_loader = data_loader

        # Save model params
        self.model_params = {
            "n_heads": n_heads,
            "emb_size": emb_size,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "data_len": len(self.data_loader.data),
        }

        super().__init__(
            num_blocks=num_blocks,
            n_heads=n_heads,
            emb_size=emb_size,
            block_size=MAX_PROTEIN_SEQ_LEN,
            vocab_size=self.data_loader.vocab_size,
            dropout=dropout,
        )
        # In a tenth of a degrees
        self.angle_emb_table = nn.Embedding(
            self.data_loader.mask_angle_idx + 1, emb_size
        )

    def forward(self, x: dict, calc_loss=True) -> torch.tensor:
        """Forward function for cermit

        Args:
            x (dict): As returned by DataLoader.generate_batch
            calc_loss (bool): Should you calc loss

        Returns:
            (None | torch.tensor): Loss if original values are given.
        """
        # Embed AA sequence and angles.
        # B, T, emb_size
        seq_embedded = self.semantic_embedding_table(x["seq_masked"])

        angles_embedded = self.angle_emb_table(x["angles_masked"]).sum(
            -2
        )  # Sum embeddings for all angles

        out, _ = self.attention_layers(
            (seq_embedded + angles_embedded, x["padding_mask"])
        )
        out = self.ln_f(out)  # B, T, emb_size
        logits = self.linear_layer(out)  # B, T, vocab_size

        B = logits.shape[0]

        loss = torch.zeros(1, device=self.data_loader.device)

        # calculate loss
        if calc_loss:
            loss_func = nn.CrossEntropyLoss()
            for batch_idx in range(B):
                # get masked index of tokens for the particular batch
                masked_idx = x["masked_idxs"][batch_idx]

                # output for that batch_index
                loss += loss_func(
                    logits[batch_idx, masked_idx],
                    x["seq_data"][batch_idx, masked_idx],
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
        if save_model:
            self.config_model_dir()

        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            batch_step = 0
            total_batches = len(self.data_loader.data) // batch_size
            data_generator = self.data_loader.generate_batch(
                batch_size=batch_size
            )
            with tqdm(
                data_generator,
                total=total_batches,
                unit=" batch",
            ) as tepoch:
                for batch_data in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    batch_step += 1
                    # get logits and loss
                    _, loss = self(batch_data)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

                    opt.step()
                    tepoch.set_postfix(loss=loss.item(), val_loss=0)

            # Save model
            if save_model:
                if epoch % checkpoint_itvl == 0:
                    self.save_model(epoch=epoch)

    def config_model_dir(
        self,
    ) -> None:
        """Config model dir. To be run once per train

        Args:
            model_name (str): Model Name. Defaults to "cermit".
        """
        if self.model_name == "cermit":
            curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
            self.model_name += f"_{curr_time}"

        self.model_dir = f"{BASE_DIR}/saved_models/{self.model_name}"

        if os.path.isdir(self.model_dir):
            # to clear the existing
            shutil.rmtree(self.model_dir)

        os.makedirs(self.model_dir)

    def save_model(self, epoch: int) -> None:
        """Save Model

        Args:
            model_path (str): Save Model
        """
        checkpoint = {"epoch": epoch}
        try:
            
            # Save all info required
            for key, val in self.model_params.items():
                checkpoint[key] = val

            checkpoint["state_dict"] = self.state_dict()
            torch.save(
                checkpoint,
                f"{self.model_dir}/{self.model_name}_epoch_{epoch}.pth",
            )

            print("Model saved successfully!")

        except Exception as err:
            print(err)

    def load_model(self, model_path: str) -> None:
        """Load saved model

        Args:
            model_path (str): Model path
        """
        try:
            self.load_state_dict(
                torch.load(model_path, map_location=self.device)["state_dict"]
            )
            print("Model loaded successfully!")
        except Exception as err:
            print(err)

    