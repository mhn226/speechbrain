#!/usr/bin/env python3
"""Recipe for fine-tuning a wav2vec model for the ST task (no transcriptions).

Author
 * Marcely Zanon Boito, 2022
"""

import sys
import torch
import logging
#import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from sacremoses import MosesDetokenizer
import speechbrain as sb
import torchaudio
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        tokens_bos, _ = batch.tokens_bos  # translation

        # wav2vec module
        feats = self.modules.wav2vec2(wavs)

        # dimensionality reduction
        src = self.modules.enc(feats)

        # transformer decoder
        #dec_out = self.modules.mBART(
        #    src, tokens_bos, pad_idx=self.hparams.pad_index
        #)
        src = self.modules.adapter(src)
        src = self.modules.length_adapter(src)
        dec_out = self.modules.mBART(
            src, tokens_bos, pad_idx=self.hparams.pad_index
        )

        # logits and softmax
        #pred = self.modules.seq_lin(dec_out.last_hidden_state)
        #p_seq = self.hparams.log_softmax(pred)
        p_seq = self.hparams.log_softmax(dec_out)
        if hparams['mbart_frozen'] and not p_seq.requires_grad:
            p_seq.requires_grad = True

        # compute outputs
        hyps = None
        if stage == sb.Stage.VALID:
            # the output of the encoder (enc) is used for valid search
            #hyps, _ = self.hparams.valid_search(src.detach(), wav_lens)
            if isinstance(self.modules.mBART, DistributedDataParallel):
                self.modules.mBART = self.modules.mBART.module

            hyps = self.modules.mBART.decode(
                    src.detach(),
                    min_decode_ratio=hparams['min_decode_ratio'],
                    max_decode_ratio=hparams['max_decode_ratio'],
                    beam_size=hparams['valid_beam_size'],
                    #eos_token_id=1000000, # big number so that the decoder doesn't stop when encoutering eos
            )
        elif stage == sb.Stage.TEST:
            #hyps, _ = self.hparams.test_search(src.detach(), wav_lens)
            if isinstance(self.modules.mBART, DistributedDataParallel):
                self.modules.mBART = self.modules.mBART.module
            hyps = self.modules.mBART.decode(
                    src.detach(),
                    min_decode_ratio=hparams['min_decode_ratio'],
                    max_decode_ratio=hparams['max_decode_ratio'],
                    beam_size=hparams['test_beam_size'],
                    #eos_token_id=hparams["eos_index"],
            )

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_seq, wav_lens, hyps,) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # st loss
        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens,)

        fr_detokenizer = MosesDetokenizer(lang=self.hparams.lang)

        if stage != sb.Stage.TRAIN:
            #predictions = [
            #    fr_detokenizer.detokenize(
            #        hparams["tokenizer"].decode_ids(utt_seq).split(" ")
            #    )
            #    for utt_seq in hyps
            #]


            detokenized_translation = [
                fr_detokenizer.detokenize(translation.split(" "))
                for translation in batch.trans
            ]
            # it needs to be a list of list due to the extend on the bleu implementation
            targets = [detokenized_translation]
            #predictions = [
            #    fr_detokenizer.detokenize(
            #        self.modules.mBART.tokenizer.batch_decode([hyp], skip_special_tokens=True)
            #    )
            #    for hyp in hyps
            #]

            #logger.info(hyps)
            #logger.info(self.modules.mBART.tokenizer.batch_decode(hyps, skip_special_tokens=True))


            predictions = [
                fr_detokenizer.detokenize(hyp.split(" ")) for hyp in self.modules.mBART.tokenizer.batch_decode(hyps, skip_special_tokens=True)
            ]


            logger.info(predictions)
            #logger.info(targets)

            self.bleu_metric.append(ids, predictions, targets)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def init_optimizers(self):
        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
        # Initializes the mbart optimizer if the model is not mbart_frozen
        if not self.hparams.mbart_frozen:
            self.mbart_optimizer = self.hparams.mbart_opt_class(
                    self.modules.mBART.parameters()
            )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()

        if self.check_gradients(loss):
            if not self.hparams.wav2vec2_frozen:  # if wav2vec2 is not frozen
                self.wav2vec_optimizer.step()
            if not self.hparams.mbart_frozen:  # if mbart is not frozen
                self.mbart_optimizer.step()
            self.adam_optimizer.step()

        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer.zero_grad()
        if not self.hparams.mbart_frozen:
            self.mbart_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.bleu_metric = self.hparams.bleu_computer()

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
            stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["BLEU"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            stats_meta = {
                "epoch": current_epoch,
                "lr_adam": old_lr_adam,
            }

            if not self.hparams.wav2vec2_frozen:
                (
                    old_lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.lr_annealing_wav2vec(stage_stats["BLEU"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
                stats_meta["lr_wav2vec"] = old_lr_wav2vec
                #self.hparams.train_logger.log_stats(
                #    stats_meta={
                #        "epoch": current_epoch,
                #        "lr_adam": old_lr_adam,
                #        "lr_wav2vec": old_lr_wav2vec,
                #    },
                #    train_stats={"loss": self.train_stats},
                #    valid_stats=stage_stats,
                #)
            if not self.hparams.mbart_frozen:
                (
                    old_lr_mbart,
                    new_lr_mbart,
                ) = self.hparams.lr_annealing_mbart(stage_stats["BLEU"])
                sb.nnet.schedulers.update_learning_rate(
                    self.mbart_optimizer, new_lr_mbart
                )
                stats_meta["lr_mbart"] = old_lr_mbart
            #else:
            #    self.hparams.train_logger.log_stats(
            #        stats_meta={"epoch": current_epoch, "lr_adam": old_lr_adam},
            #        train_stats={"loss": self.train_stats},
            #        valid_stats=stage_stats,
            #    )
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # create checkpoing
            meta = {"BLEU": stage_stats["BLEU"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, max_keys=["BLEU"]
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "trans", "tokens_list", "tokens_bos", "tokens_eos",
    )
    def reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(translation, return_tensors="pt")
        tokens_list = labels['input_ids'].tolist()[-1][1:-2]
        #tokens_list = hparams["tokenizer"].encode_as_ids(translation)
        #logger.info(tokens_list)
        yield tokens_list
        #tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        tokens_bos = torch.LongTensor([tokenizer.lang_code_to_id["fr_XX"]] + (tokens_list))
        #logger.info(tokens_bos)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        #logger.info(tokens_eos)
        yield tokens_eos

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "valid"]:
        csv_path = f"{data_folder}/{dataset}.csv"

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline_func, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "trans",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
            ],
        )

    for dataset in ["test"]:
        csv_path = f"{data_folder}/{dataset}.csv"
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "trans",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 3},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 3},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration"
            )

        hparams["dataloader_options"]["shuffle"] = False
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 3},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 3},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["dataloader_options"]["shuffle"] = False
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 3},
                sort_key="duration",
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1}, key_max_value={"duration": 3},
            )

        hparams["dataloader_options"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Fetch pretrained modules
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, st_brain.modules.mBART.tokenizer)

    # Before training, we drop some of the wav2vec 2.0 Transformer Encoder layers
    st_brain.modules.wav2vec2.model.encoder.layers = st_brain.modules.wav2vec2.model.encoder.layers[
        : hparams["keep_n_layers"]
    ]


    print(st_brain.modules.mBART)

    # Training
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    for dataset in ["valid", "test"]:
        st_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test" + ".txt"  
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
