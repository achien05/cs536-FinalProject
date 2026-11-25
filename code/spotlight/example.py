import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import joblib
import sys
import time
import argparse

from datasets.movielens import get_movielens_dataset
from interactions import *
from collections import defaultdict
from factorization.explicit import ExplicitFactorizationModel
from cross_validation import *
from evaluation import rmse_score

from sklearn.utils import murmurhash3_32
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from factorization.representations import BilinearNet

parser = argparse.ArgumentParser()
parser.add_argument('--optim', choices=["Adam", "GD"], default="GD")
parser.add_argument('--edim', type=int, default=64)
parser.add_argument('--batch_s', type=int, default=128)
parser.add_argument('--niter', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ilam', type=float, default=0.1)
parser.add_argument('--ulam', type=float, default=0.0001)
parser.add_argument('--test', type=float, default=0.2)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not os.path.exists(os.getcwd() + "/Mlens_100K.joblib"):
    dataset_100k = get_movielens_dataset(variant='100K')
    joblib.dump(dataset_100k, "Mlens_100K.joblib")
else:
    dataset_100k = joblib.load(os.getcwd() + "/Mlens_100K.joblib")

if args.optim == "Adam":
    model = ExplicitFactorizationModel(loss='regression',
            embedding_dim=args.edim,
            n_iter=args.niter, 
            batch_size=args.batch_s,
            l2=0,
            learning_rate=args.lr,
            sep_penalty=True,
            item_penlam=args.ilam,
            user_penlam=args.ulam,                 
            jnt_penalty=False,
            all_penlam=0,
            optimizer_func=torch.optim.Adam,
            lr_sched=True,
            use_cuda=torch.cuda.is_available())
else:
    model = ExplicitFactorizationModel(loss='regression',
            embedding_dim=args.edim,
            n_iter=args.niter, 
            batch_size=args.batch_s,
            l2=0,
            learning_rate=args.lr,
            sep_penalty=True,
            item_penlam=args.ilam,
            user_penlam=args.ulam,                 
            jnt_penalty=False,
            all_penlam=0,
            optimizer_func=torch.optim.SGD,
            lr_sched=True,
            use_cuda=torch.cuda.is_available())

train, test = random_train_test_split(dataset_100k, args.test)

model.fit(train, test, verbose=False)

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)
print('final trainRMSE {:.3f}, final testRMSE {:.3f}'.format(train_rmse, test_rmse))
