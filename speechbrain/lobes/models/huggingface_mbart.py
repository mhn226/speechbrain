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

# We check if transformers is installed.
try:
    import transformers
    from transformers import MBartModel, MBartConfig
    from transformers import MBartForConditionalGeneration, MBartTokenizer
    from transformers import MBart50TokenizerFast, MBart50Tokenizer
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)

HF_models = {
    "mbart": MBartForConditionalGeneration,
    #"mbart": MBartModel,
}

HF_config = {
    "mbart": MBartConfig,
}


class HuggingFaceMBART(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
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
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    target_lang: str (default: fr_XX (a.k.a French)
        The target language code according to mbart model
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
        #output_norm=True,
        freeze=True,
        target_lang="fr_XX",
    ):
        super().__init__()


        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        #if "hubert" in source:
        #    config = HF_config.get("hubert")
        #    model = HF_models.get("hubert")
        #elif "wavlm" in source:
        #    config = HF_config.get("wavlm")
        #    model = HF_models.get("wavlm")
        #else:
        #    config = HF_config.get("wav2vec2")
        #    model = HF_models.get("wav2vec2")
     

        # hard code max_length=2500
        #max_length = 2500
        # hard code d_model=, tgt_vocab=250054
        #tgt_vocab = 250054
        #d_model = 1024
        #self.attention_type = "regularMHA"
        #self.positional_encoding_type = "fixed_abs_sine"
        #if self.positional_encoding_type == "fixed_abs_sine":
        #    self.positional_encoding = PositionalEncoding(d_model, max_length)
        #elif positional_encoding is None:
        #    pass
        #    # no positional encodings
        #self.custom_tgt_module = ModuleList(
        #    NormalizedEmbedding(d_model, tgt_vocab)
        #)

        config = HF_config.get("mbart")
        model = HF_models.get("mbart")

        self.target_lang = target_lang
        # Download and load the model
        self._from_pretrained(
            source, config=config, model=model, save_path=save_path
        )
        self.freeze = freeze
        #self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_mbart - mbart is frozen."
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
            self._load_sb_pretrained_mbart_parameters(ckpt_full_path)
        else:
            #self.tokenizer = MBart50TokenizerFast.from_pretrained("mbart-large-50/", tgt_lang=self.target_lang)
            self.tokenizer = MBart50Tokenizer.from_pretrained("mbart-large-50/", tgt_lang=self.target_lang)
            #self.model = model.from_pretrained(source, cache_dir=save_path).get_decoder()
            self.model = model.from_pretrained(source, cache_dir=save_path)
            self.model.model.encoder = nn.Identity()
            #self.model.encoder = nn.Identity()
            
            self.model.config.forced_eos_token_id = self.tokenizer.lang_code_to_id[self.target_lang]
            #self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.target_lang]
            # Remove embed layers
            #self.model.embed_tokens = nn.Identity()
            #self.model.embed_positions = nn.Identity()

    def _load_sb_pretrained_mbart_parameters(self, path):
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
            if "mbart." in key:
                save_key = key.replace("model.mbart.", "")
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

    def forward_bk(self, src, tgt):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        src : torch.Tensor (output from enc)
        tgt : target tokens
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                #return self.model(inputs_embeds=src, decoder_input_ids=tgt)
                return self.model(src, tgt)
        
        #return self.model(inputs_embeds=src, decoder_input_ids=tgt)
        return self.model(src, tgt)

    def forward(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task using a wav2vec encoder
        (same than above, but without the encoder stack)
    
        Arguments
        ----------
        src (transcription): tensor
            output features from the w2v2 encoder
        tgt (translation): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        # Transform encoder's output to the right format of the MBartModel
        
        # should we replace 0 elements by pax_idx as pad_idx of mbart model seems to be different from 0?
        tgt = self.custom_padding(tgt, 0, pad_idx)

        #(
        #    src_key_padding_mask,
        #    tgt_key_padding_mask,
        #    src_mask,
        #    tgt_mask,
        #) = self.make_masks_for_mt(src, tgt, pad_idx=pad_idx)
        #print(tgt)
        #print(self.model.config.pad_token_id, self.model.config.eos_token_id, pad_idx, self.model.config.decoder_start_token_id)
        #print(src_mask, src_key_padding_mask)
        #print(tgt_mask, tgt_key_padding_mask)

        
        src = BaseModelOutput(last_hidden_state=src)

        if self.freeze:
            with torch.no_grad():
                dec_out = self.model(decoder_input_ids=tgt, encoder_outputs=src).logits.detach()
                return dec_out
        
        #dec_out = self.model(input_ids=tgt, attention_mask=None, encoder_hidden_states=src)
        dec_out = self.model(decoder_input_ids=tgt, encoder_outputs=src).logits
        
        return dec_out


    @torch.no_grad()
    def decode(
            self,
            encoder_out,
            min_decode_ratio=0.0,
            max_decode_ratio=1.0,
            beam_size=5,
            eos_token_id=2,
    ):
        """This method implements a decoding step for the transformer model.
        
        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.
        """
        encoder_len = encoder_out.size(1)
        min_len = int(encoder_len * min_decode_ratio)
        max_len = int(encoder_len * max_decode_ratio)

        encoder_out = BaseModelOutput(last_hidden_state=encoder_out)


        dec_out = self.model.generate(
            #inputs=encoder_out,
            encoder_outputs=encoder_out,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
            min_length=min_len,
            max_length=max_len,
            num_beams=beam_size,
            #eos_token_id=eos_token_id,
        )
        
        return dec_out

    def custom_padding(self, x, org_pad, custom_pad):
        out = x.clone()
        out[x==org_pad] = custom_pad

        return out

    def make_masks_for_mt(self, src, tgt, pad_idx=0):
        """This method generates the masks for training the transformer model.
                
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if self.training:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)

        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask




