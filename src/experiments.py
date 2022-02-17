# Experiments with different parameters for training, etc.
from train_bbp import Trainer
from config import CFG

CFG.n_samples_train = 0
CFG.n_samples_val = 500
trainer = Trainer('C:/Users/jhazelde/BNBP-dev/src/pytorch_impl/HH_2000_model_200_I0_1.5_0.100000.pt')
trainer.validate()