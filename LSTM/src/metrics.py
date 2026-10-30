"""Binary risk classification metrics with small-sample edge cases."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, roc_curve, log_loss)


def binary_metrics(target, probabilities, threshold=0.5):
    y = np.asarray(target, np.int64)
    p = np.asarray(probabilities, np.float64)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('指标输入数量或数值异常')
    if np.min(p) < 0 or np.max(p) > 1:
        raise ValueError('模型概率必须在[0,1]')
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    result = {'n_companies': len(y), 'n_risk': int(sum(y==1)),
              'threshold': float(threshold),
              'accuracy': float(accuracy_score(y,pred)),
              'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
              'precision_risk': float(precision_score(y,pred,zero_division=0)),
              'recall_risk': float(recall_score(y,pred,zero_division=0)),
              'f1_risk': float(f1_score(y,pred,zero_division=0)),
              'macro_f1': float(f1_score(y,pred,average='macro',zero_division=0)),
              'micro_f1': float(f1_score(y,pred,average='micro',zero_division=0)),
              'specificity': float(tn/max(1,tn+fp)),
              'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        result.update(auroc=float(roc_auc_score(y,p)),
                      pr_auc=float(average_precision_score(y,p)),
                      ks=float(np.max(tpr-fpr)),
                      binary_log_loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])))
    else:
        result.update(auroc=None, pr_auc=None, ks=None, binary_log_loss=None)
    return result
