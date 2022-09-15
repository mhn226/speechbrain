"""This lobe enables the integration of huggingface pretrained wav2vec2/hubert/wavlm models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
"""

import os
import torch
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch import nn
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch
from typing import Optional
from speechbrain.nnet.containers import ModuleList
from speechbrain.dataio.dataio import length_to_mask

from speechbrain.lobes.models.transformer.TransformerST import TransformerST
from speechbrain.lobes.models.transformer.Transformer import (
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
    PositionalEncoding,
    #TransformerDecoder,
    #TransformerEncoder,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# We check if transformers is installed.
try:
    import transformers
    from transformers import BertModel, BertTokenizerFast, BertConfig
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)

HF_models = {
    "labse": BertModel,
}

HF_config = {
    "labse": BertConfig,
}


class HuggingFaceLaBSE(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained LaBSE models.

    Source paper LaBSE: https://arxiv.org/abs/2007.01852
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
    ):
        super().__init__()


        config = HF_config.get("labse")
        model = HF_models.get("labse")

        # Download and load the model
        self._from_pretrained(
            source, config=config, model=model, save_path=save_path
        )
        self.freeze = freeze
        #self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_labse - labse is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def _from_pretrained(self, source, config, model, save_path):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """

        is_sb, ckpt_file = self._check_model_source(source)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file, source=source, savedir=save_path
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_labse_parameters(ckpt_full_path)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
            self.model = model.from_pretrained(source, cache_dir=save_path)
            

    def _load_sb_pretrained_labse_parameters(self, path):
        """Loads the parameter of a w2v2 model pretrained with SpeechBrain and the
        HuggingFaceWav2Vec2Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility between HuggingFaceWav2Vec2Pretrain
        and HuggingFaceWav2Vec2.

        In practice a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """

        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            if "labse." in key:
                save_key = key.replace("model.labse.", "")
                modified_state_dict[save_key] = params

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for wav2vec 2.0 finetuning."
            )

    def _check_model_source(self, path):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True
        is_sb = True

        # If path is a huggingface hub.
        if not source.exists():
            is_local = False

        if is_local:
            # Test for HuggingFace model
            if any(File.endswith(".bin") for File in os.listdir(path)):
                is_sb = False
                return is_sb, checkpoint_filename

            # Test for SpeechBrain model and get the filename.
            for File in os.listdir(path):
                if File.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, File)
                    is_sb = True
                    return is_sb, checkpoint_filename
        else:
            files = model_info(
                path
            ).siblings  # get the list of files of the Hub

            # Test if it's an HuggingFace model or a SB one
            for File in files:
                if File.rfilename.endswith(".ckpt"):
                    checkpoint_filename = File.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename

            for File in files:
                if File.rfilename.endswith(".bin"):
                    checkpoint_filename = File.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)


    def forward(self, input_texts):
        """This method implements a forward step for mt task using a wav2vec encoder
        (same than above, but without the encoder stack)
    
        Arguments
        ----------
        input_texts (translation): list
            The list of texts (required).
        """
        # Transform encoder's output to the right format of the LaBSE model
        if self.freeze:
            with torch.no_grad():
                input_texts = self.tokenizer(input_texts, return_tensors="pt", padding=True)
                for key in input_texts.keys():
                    input_texts[key] = input_texts[key].cuda(device=self.model.device)
                    input_texts[key].requires_grad = False
                
                embeddings = self.model(**input_texts).pooler_output
                return embeddings

        input_texts = self.tokenizer(input_texts, return_tensors="pt", padding=True)
        embeddings = self.model(**input_texts).pooler_output

        return embeddings

