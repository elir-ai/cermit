import pandas as pd
import torch

MASK_TOKEN = "[MASK]"


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

        vocab = [MASK_TOKEN] + list(set(unique_chars))
        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        self.mask_token_idx = self.char_to_ix[MASK_TOKEN]

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
        self, split_type, batch_size, train_split=0.8, masking_percent=0.15
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
        original, masked, masked_idxs = self.mask_sequences(original)
        return original, masked, masked_idxs
