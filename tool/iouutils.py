import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    ##  The 5-th cls is exclude.
    n_class = n_class+1
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        lp[lt==4]=4
        tmp = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += tmp
    hist[4,4]=0
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist)[0:4] / hist.sum(axis=1)[0:4]
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist)[0:4] / ((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:4])
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1)[0:4] / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    dice_scores = {}
    for i in range(4):  # Only compute for the first 4 classes
        tp = np.diag(hist)[i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        dice_scores[i] = dice

    mean_dice = np.mean(list(dice_scores.values()))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "Dice Coefficients": dice_scores,
        "Mean Dice": mean_dice
    }

