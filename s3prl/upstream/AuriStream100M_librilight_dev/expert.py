from collections import OrderedDict
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import (AutoConfig, AutoModel)

SAMPLE_RATE = 16000

class UpstreamExpert(nn.Module):
    def __init__(self, name='', **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "AuriStream100M_librilight_dev"

        self.extracter = AutoModel.from_pretrained(
            'TuKoResearch/WavCochV8192', trust_remote_code=True) # Add eval() ?
        self.model = AutoModel.from_pretrained(
            'TuKoResearch/AuriStream100M_librilight_dev', trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(
            'TuKoResearch/AuriStream100M_librilight_dev', trust_remote_code=True)


    def get_downsample_rates(self) -> float:
        return 80.2 # HuBERT, wav2vec etc uses 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        Modeled after HuBERT, wav2vec etc.
        """

        device = wavs[0].device

        # [B, T] → pad
        wavs = [wav if wav.ndim == 1 else wav.squeeze() for wav in wavs]
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)  # [B, T]
        # Should we do anything with the attention mask?

        # Add channel dim → [B, 1, T]
        wavs = wavs.unsqueeze(1)

        # print(f'\n\nLen wavs: {len(wavs)}')
        print(f"Wav shape: {wavs.shape}")  # should be [B, 1, T]

        # print(f'\n\nLen wavs: {len(wavs)}')
        # # print size of each wav
        # for wav in wavs:
        #     print(f'Wav shape: {wav.shape}')

        # Get codes
        input_values = self.extracter(
            wavs,
            # padding=True,
            sample_rate=SAMPLE_RATE,
        ).to(device)
        output_values = self.model(
            input_values.input_values,
            output_hidden_states=True,
            # return_dict=True
        )

        return {"hidden_states": output_values.hidden_states}
