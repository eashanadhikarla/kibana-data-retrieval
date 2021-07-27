# accuracy computation
def accuracy(model, ds, pct):
    # assumes model.eval()
    # percent correct within pct of true pacing rate
    n_correct = 0; n_wrong = 0

    for i in range(len(ds)):
        (X, Y) = ds[i]                # (predictors, target)
        X, Y = X.float(), Y.float()
        with torch.no_grad():
            output, _, _, _ = model(X)         # computed price

        abs_delta = np.abs(output.item() - Y.item())
        max_allow = np.abs(pct * Y.item())
        if abs_delta < max_allow:
            n_correct +=1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc*100

