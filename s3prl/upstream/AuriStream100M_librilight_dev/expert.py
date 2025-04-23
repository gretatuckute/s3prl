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


    def get_downsample_rates(self, key: str = None) -> int:
        return 1 # HuBERT, wav2vec etc uses 320

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

        # Add channel dim → [B, 1, T]
        wavs = wavs.unsqueeze(1)

        print(f'\n\nLen wavs: {len(wavs)}')
        print(f"Wav shape: {wavs.shape}")  # should be [B, 1, T]

        print(f'\n\nLen wavs: {len(wavs)}')
        # print size of each wav
        for wav in wavs:
            print(f'Wav shape: {wav.shape}')


        # Get codes
        input_values = self.extracter(
            wavs,
            # padding=True,
            sample_rate=SAMPLE_RATE,
        ).to(device)
        # add padding?
        output_values = self.model(
            input_values.input_values,
            output_hidden_states=True,
            # return_dict=True
        )

        return {"hidden_states": output_values.hidden_states}

        # For debugging, let's truncate to the first 2000 samples assuming we have a tensor
        # wavs = [wav[:2000] for wav in wavs]

        # 2. Pad to max length → [B, T]
        # wavs_tensor = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        #
        # # 3. Add channel dim → [B, 1, T]
        # wavs_tensor = wavs_tensor.unsqueeze(1).to(device)
        #
        # print(f"[forward] Batch shape: {wavs_tensor.shape}")  # should be [B, 1, T]

        # all_outputs = []
        #
        # for wav in wavs:
        #     # Make sure each wav has shape [1, 1, T]
        #     if wav.ndim == 1:
        #         wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        #     elif wav.ndim == 2:
        #         wav = wav.unsqueeze(0)  # [1, 1, T]
        #     # else assume it's already [1, 1, T]
        #
        #     wav = wav.to(device)
        #
        #     # Extract codes (pass through extracter)
        #     input_values = self.extracter(
        #         wav,
        #         sample_rate=SAMPLE_RATE,
        #     ).to(device)
        #
        #     # Model forward
        #     output = self.model(
        #         input_values.input_values,
        #         output_hidden_states=True
        #     )
        #
        #     all_outputs.append(output)

        # return {
        #     # "hidden_states": [o.hidden_states for o in all_outputs],
        #     # "last_hidden_state": [o.last_hidden_state for o in all_outputs],
        # }



        # MIN_LEN = 16000  # 1 second minimum length
        # wavs = [F.pad(wav, (0, max(0, MIN_LEN - wav.shape[-1]))) for wav in wavs]

        # Pad waveforms to same length, get [B, T]
        # wavs_tensor = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)

        # Add channel dimension → [B, 1, T]
        # wavs_tensor = wavs_tensor.unsqueeze(1)

        # wavs = [wav.detach().cpu().numpy() for wav in wavs]


        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        # return {
        #     "hidden_states": [hidden, feature],
        #     "PR": [hidden, feature], # perhaps get rid of everything beyond hidden_states ?
        #     "ASR": [hidden, feature],
        #     "QbE": [hidden, feature],
        #     "SID": [hidden, feature],
        #     "ASV": [hidden, feature],
        #     "SD": [hidden, feature],
        #     "ER": [hidden, feature],
        #     "SF": [hidden, feature],
        #     "SE": [hidden, feature],
        #     "SS": [hidden, feature],
        #     "secret": [hidden, feature],
        # }
