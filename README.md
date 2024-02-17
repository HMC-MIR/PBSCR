# PBSCSR
Piano Bootleg Score Composer Style Recognition - Dataset &amp; Baselines

The purpose of this repository is to recognize the composer of a piece based on the bootleg score, which contains only notehead positions on the staff. This repository contains labeled data, unlabeled data for pretraining, and the code necessary to reproduce the results.

To reproduce the GPT2 results:
1. Create a Python virtual environment using baselines.yml
2. Run 01_gpt2_pretraining.ipynb to generate the bash script for pretraining
3. Unzip 9_way_dataset.zip and 100_way_dataset.zip
4. Run pretrain_lm.sh to pretrain the model
5. Run gpt2_LP_and_FT.ipynb
