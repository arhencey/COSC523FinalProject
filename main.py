# This file implements all the different models and prints the
# RMSE values after training.
#
# Author: Alan Hencey

import argparse
import sys

from Baselinetfidfcomplexity import basetfidf
from SVMwithextracfeats import svmfeats
from SVMplusElmo import svmelmo
from models import train_features_only, train_excerpts, train_rnn

parser = argparse.ArgumentParser()
#parser.add_argument("--run_all", help="run all of the models and print out their RMSE scores", action="store_true")
parser.add_argument("--basetfidf", help="run the baseline TF-IDF model on the dataset", action="store_true")
parser.add_argument("--svm_features", help="run the SVM model on the engineered features", action="store_true")
parser.add_argument("--svm_elmo", help="run the SVM model with ELMO embeddings on the excerpts", action="store_true")
parser.add_argument("--nn_features", help="run the feedforward nerual net on the engineered features", action="store_true")
parser.add_argument("--nn_excerpts", help="run the feedforward neural net on the excerpts", action="store_true")
parser.add_argument("--rnn_excerpts", help="run the RNN on the excerpts", action="store_true")
args = parser.parse_args()

def run_all():
    baseRMSE = basetfidf()
    svmfeatsRMSE = svmfeats()
    svmelmoRMSE = svmelmo()
    nnfeatsRMSE = train_features_only()
    nnexcerptsRMSE = train_excerpts()
    rnnRMSE = train_rnn()

    print("              RMSE RESULTS                 ")
    print("------------------------------------------------")
    print(f"Baseline TF-IDF:              {baseRMSE}")
    print("------------------------------------------------")
    print(f"SVM with engineered features: {svmfeatsRMSE}")
    print("------------------------------------------------")
    print(f"SVM with ELMO embeddings:     {svmelmoRMSE}")
    print("------------------------------------------------")
    print(f"NN with engineered features:  {nnfeatsRMSE}")
    print("------------------------------------------------")
    print(f"NN with excerpts:             {nnexcerptsRMSE}")
    print("------------------------------------------------")
    print(f"RNN:                          {rnnRMSE}")
    print(f"-----------------------------------------------")

def main():
    if not len(sys.argv) > 1:
        run_all()
    elif args.basetfidf:
        print("Running Baseline TF-IDF model...")
        rmse = basetfidf()
        print(f"Baseline TF-IDF RMSE: {rmse}")
    elif args.svm_features:
        print("Running SVM with engineered features...")
        rmse = svmfeats()
        print(f"SVM with engineered features RMSE: {rmse}")
    elif args.svm_elmo:
        print("Running SVM with ELMO embeddings...")
        rmse = svmelmo()
        print(f"SVM with ELMO embeddings RMSE: {rmse}")
    elif args.nn_features:
        print("Running NN with engineered features...")
        rmse = train_features_only()
        print(f"NN with engineered features RMSE: {rmse}")
    elif args.nn_excerpts:
        print("Running NN with excerpts...")
        rmse = train_excerpts()
        print(f"NN with excerpts RMSE: {rmse}")
    elif args.rnn_excerpts:
        print("Running RNN...")
        rmse = train_rnn()
        print(f"RNN RMSE: {rmse}")
    else:
        print("Invalid arguments.")
        quit()


if __name__=='__main__':
    main()
