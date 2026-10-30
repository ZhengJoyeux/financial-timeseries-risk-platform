"""Read best real-validation epoch for each training strategy, without opening test."""
import csv
from pathlib import Path

MODES=('real_only','generated_only','real_plus_s1')

def main():
    for mode in MODES:
        p=Path('LSTM/outputs/logs')/f'{mode}_s1_2027'/'history.csv'
        if not p.is_file():
            print(mode,'未完成正式训练：',p)
            continue
        with p.open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        row=min(rows,key=lambda r:float(r['val_macro_bce']))
        print(mode,'best_epoch=',row['epoch'], 'val_macro_bce=',row['val_macro_bce'],
              'val_PR_AUC=',row['val_pr_auc'],'val_macro_F1=',row['val_macro_f1'],
              'val_bal_acc=',row['val_bal_acc'])

if __name__=='__main__':main()
