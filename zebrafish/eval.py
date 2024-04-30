import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing

def linear_probe(model: nn.Module, ds: torch.utils.data.Dataset):
    # extract feature vectors in batches
    orig_training = model.training
    model.eval()
    fvec_batches = []
    label_batches = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(ds, 512):
            fvec_batches.append(model(batch['image']).numpy())
            label_batches.append(batch['label'].numpy())
            break
    fvecs = np.concatenate(fvec_batches)
    labels = np.concatenate(label_batches)
    model.train(orig_training)

    # l2 regularized logistic regression optimized w/ cross-validation
    rng = np.random.default_rng()
    shuffle_idx = rng.permutation(fvecs.shape[0])
    X = sklearn.preprocessing.StandardScaler().fit_transform(fvecs[shuffle_idx,:])
    y = labels[shuffle_idx]
    clf = sklearn.linear_model.LogisticRegressionCV(Cs=10, penalty='l2', solver='lbfgs', scoring='balanced_accuracy', max_iter=1000)
    clf.fit(X, y)
    # max metric averaged across folds
    best = clf.scores_[0].mean(axis=0).max() 
    return best