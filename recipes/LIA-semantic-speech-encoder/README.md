# Spoken Dialogue State Tracking

This recipe presents how to reproduce the training of [SAMU-XLSR](https://arxiv.org/abs/2205.08180).

## General overview

Traditionally, Task-Oriented Dialogue systems update their understanding of the user's needs in three steps: transcription of the user's utterance, semantic extraction of the key concepts, and contextualization with the previously identified concepts. Such cascade approaches suffer from cascading errors and separate optimization. End-to-End approaches have been proved helpful up to the semantic extraction step. This repository attempts to go one step further paving the path towards completely neural spoken dialogue state tracking by comparing three approaches: (1) a state of the art cascade approach, (2) a locally E2E approach with rule-based contextualization and (3) a completely neural approach.

## Getting started

- [ ] Create a new conda environment with `conda create -n myenv python=3.10`.
- [ ] Enter your fresh environment with `conda activate myenv`.
- [ ] Run `pip install -r requirements.txt`.
- [ ] Download the IWSLT Tamasheq dataset with `git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git`.
- [ ] Clone this repository with `git clone https://github.com/mhn226/speechbrain.git`.
- [ ] Switch to the branch with this recipe with `git checkout semantic-speech-enc`.
- [ ] Run `cd speechbrain && pip install -e .`.
- [ ] Move to the recipe's folder with `cd recipes/LIA-semantic-speech-encoder`.
- [ ] Pre-compute the LABSE embeddings with `python precompute_semantic_embeddings.py --data_csv_folder ../../../IWSLT2022_Tamasheq_data/taq_fra_clean/csv_version/ --language french --batch_size 8 --model_type labse --word_col_name trans`.
- [ ] Run the training with `python train_lia_semantic_speech_encoder.py hparams/train_lia_semantic_speech_encoder.yaml --root_data_folder ../../../IWSLT2022_Tamasheq_data/taq_fra_clean`.

## More detailed understanding

### Precomputing the textual semantic embeddings

The script `precompute_semantic_embeddings.py` considers all the csv files in the provided `data_csv_folder` and creates (in the same folder) a `{model_type}_embeddings.ark` file which contains the embeddings and a `{model_type}_embeddings.scp` which associates each example id to a portion of the ark file. Careful because the path to the ark file present in the scp file is an absolute path dependant on the file system you are using. If you want to change the path to the ark file without recomputing the embeddings you can do it with `python change_ark_path.py --scp_file ../../../IWSLT2022_Tamasheq_data/taq_fra_clean/csv_version/labse_embeddings.scp --new_ark_path ../../../IWSLT2022_Tamasheq_data/taq_fra_clean/csv_version/labse_embeddings.ark` which will put relative path instead of absolute ones for instance.