# PBSCR
Piano Bootleg Score Composer Recognition - Dataset &amp; Baselines

The purpose of this repository is to recognize the composer of a piece based on the bootleg score, which contains only notehead positions on the staff. This repository contains labeled data, unlabeled data for pretraining, and the code necessary to reproduce the results.

See below for instructions on data replication.

## GPT2
1. Activate baselines virtual environment: conda activate baselines
2. Run the preprocessing notebook located in /home/username/ttmp/PBSCR/baselines/LM_pretraining_data_preprocessing.ipynb
3. Run pretraining notebook located in /home/username/ttmp/PBSCR/baselines/01_gpt2_pretraining.ipynb to create bash script for pretraining
4. Run the pretraining bash script located at /home/username/ttmp/PBSCR_data/pretrained_model/pretrain_lm.sh
5. Run the jupyter notebook for the GPT2 notebook at gpt2_LP_and_FT.ipynb

## CNN
1. Activate baselines virtual environment: conda activate baselines
2. Git clone PBSCR repo into ttmp
3. CNN jupyter notebook is located in /home/username/ttmp/PBSCR/baselines/CNN/simple_CNN.ipynb
4. Specific data replication instructions for 9_way_dataset and 100_way_dataset are in the simple_CNN.ipynb
5. CNN does not require language modeling (no need to run LM_pretraining_data_preprocessing.ipynb)

## RoBERTa
1. Activate baselines virtual environment: conda activate baselines
2. Git clone PBSCR repo into ttmp (if done before, skip this step)
3. (Optional) Run  data_creation.ipynb
   This jupyter notebook clone the imslp_bootleg_dir-v1
   Filter filler, which generate imslp_bootleg_dir-v1.1
   This allows you to have a copy of both versions of imslp bootleg.
6. The PBSCR repo already has imslp_bootleg_dir-v1.1, you can just point to this directory
7. Run LM_pretraining_data_preprocessing.ipynb,which is located in /PBSCR/baselines/LM_pretraining_data_preprocessing.ipynb. Does not matter which version of  imslp bootleg you are pointing to in this step because there’s a same filter filler like in step 3 to generate imslp_bootleg_dir-v1.1. (if )

8. Run 01_roberta_pretraining.ipynb stop after finishing running Language Model Pretraining section.
9. Before keep going to run Language Model Pretraining Curves section, run the bash script  train_lm.sh in the output directory you specify when running Language Model Pretraining
10. Run the bash script in a persistent shell session (like tmux or screen) with the baselines environment. This process may take about 4-5 hours
11. After finish running train_lm.sh , you can run the  Language Model Pretraining Curves section
12. You should see that the train data curve (black) is similar to validation data curve (green)
13. Run roberta/roberta_LP_and_FT.ipynb

## Few-Shot
1. Activate baselines virtual environment: conda activate baselines
2. Run the embeddings notebook located at /home/username/ttmp/PBSCR/baselines/fewshot_embeddings.ipynb
3. Run the experiment notebook located at /home/username/ttmp/PBSCR/baselines/fewshot_experiment.ipynb


## Citation

Arhan Jain, Alec Bunn, Austin Pham, and TJ Tsai.  "PBSCR: The Piano Bootleg Score Composer Recognition Dataset."  Transactions of the International Society for Music Information Retrieval, to appear.


## Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. 2144050.  Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.


