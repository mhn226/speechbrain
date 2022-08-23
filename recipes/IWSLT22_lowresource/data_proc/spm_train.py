import sentencepiece as spm

data_dir='/home/getalp/nguyen35/speechbrain/recipes/IWSLT22_lowresource/data_proc/IWSLT2022_Tamasheq_data/taq_fra_clean/train/'


spm.SentencePieceTrainer.train(input=data_dir + 'txt/train.fra', model_prefix=data_dir + 'spm_unigram1000', vocab_size=1000, model_type='unigram')

