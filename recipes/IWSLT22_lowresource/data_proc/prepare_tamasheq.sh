#!/bin/bash
# ############################################################################
# Tamasheq-French data processing pipeline
#
# Requirements: python, git, sentencepiece command line extension
# Author:  Marcely Zanon Boito, 2022
# ############################################################################

#cd data_proc/

# 1. clone the IWSLT 2022 Tamasheq-French dataset
#git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git

data_dir=/home/getalp/nguyen35/speechbrain/recipes/IWSLT22_lowresource/data_proc


# 2. train the tokenizer
# /!\ it requires the command line extension for sentence piece, available here: https://github.com/google/sentencepiece
#spm_train --input $data_dir/IWSLT2022_Tamasheq_data/taq_fra_clean/train/txt/train.fra  --vocab_size=1000 --model_type=unigram --model_prefix=$data_dir/IWSLT2022_Tamasheq_data/taq_fra_clean/train/spm_unigram1000

python spm_train.py

# 3. generate json files for the speechbrain recipe
mkdir -p $data_dir/IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/
python to_json.py $data_dir/IWSLT2022_Tamasheq_data/taq_fra_clean/ $data_dir/IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/
