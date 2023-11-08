# from fastai import *
# from fastai.text import *
from fastai.text.all import *

import glob
import eval_models

bs=48


torch.cuda.set_device(0)


path = "pretraining_files/chunked"

texts = get_files(path, extensions=[".txt"], folders=["train", "valid"])
print(len(texts))
def read_file(f): return L(f.read_text().split(' '))

splits = RandomSplitter(valid_pct=0.1)(texts)
tfms = [Tokenizer.from_folder(path), Numericalize()]
dsets = Datasets(texts, [tfms], splits=splits, dl_type=LMDataLoader)


bs,sl=48,64
dbunch_lm = dsets.dataloaders(bs=bs, seq_len=sl, val_bs=bs, device=torch.device('cuda'))


dbunch_lm.show_batch()


learn = language_model_learner(dbunch_lm, AWD_LSTM, drop_mult=0.5, pretrained=False)


lr = 3e-3

torch.cuda.set_device('cuda:0')

# Explicitly set model to cuda
# learn.model = learn.model.cuda()
learn.model = learn.model.cuda(0)
# print(learn.model)

# print(learn.model.device)

learn.unfreeze()
learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8))

learn.save("models/awdlstm_imslp_wt", with_opt=False)
learn.data.vocab.save("models/awdlstm_imslp_vocab.pkl")