"""One-time final test evaluation, called only after all 3 runs are frozen."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from VAE.src.dataset import FinancialScaler
from LSTM.src.common import physical_gpu1, sha256_file, json_save_exclusive
from LSTM.src.data import read_real_split, load_npz
from LSTM.src.metrics import binary_metrics
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.scripts.train import MODES, inference


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='LSTM/config/multihead_lstm.yaml')
    ap.add_argument('--output',default='LSTM/outputs/evaluation/s1_medianfix_seed2027_comparison')
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'已有正式测试结果，拒绝再次运行/覆盖：{output}')
    device=physical_gpu1()
    datadir=Path(cfg['project']['dataset'])
    manifest_path=datadir/'manifest.json'
    preprocess_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    train_ids, _, _=read_real_split('data/splits/train.xlsx',
                   json.loads(Path('data/splits/split_manifest.json').read_text()), 'train')
    val_ids,_,_=read_real_split('data/splits/validation.xlsx',
                   json.loads(Path('data/splits/split_manifest.json').read_text()),'validation')
    records=[]
    checkpoints={}
    for mode in MODES:
        file=Path('LSTM/outputs/checkpoints')/f'{mode}_s1_2027'/'best.pt'
        if not file.is_file():
            raise FileNotFoundError(f'三组模型必须先训练完成，缺少：{file}')
        ckpt=torch.load(file,map_location='cpu',weights_only=False)
        if ckpt.get('mode')!=mode or ckpt.get('test_used') is not False:
            raise RuntimeError(f'{file}模型模式或测试来源错误')
        if ckpt['manifest_sha256']!=sha256_file(manifest_path):
            raise RuntimeError(f'{file}预处理manifest不匹配')
        if ckpt['train_npz_sha256']!=sha256_file(datadir/f'train_{mode}.npz'):
            raise RuntimeError(f'{file}训练集文件发生变化')
        if ckpt['val_npz_sha256']!=sha256_file(datadir/'validation.npz'):
            raise RuntimeError(f'{file}验证集文件发生变化')
        checkpoints[mode]=(file,ckpt)
    split_manifest=json.loads(Path('data/splits/split_manifest.json').read_text(encoding='utf-8'))
    test_ids,test_raw,test_labels=read_real_split('data/splits/test.xlsx',split_manifest,'test')
    assert not(set(test_ids)&(set(train_ids)|set(val_ids))), '企业级数据泄漏'
    y_test=np.asarray([int(v=='有风险') for v in test_labels],dtype=np.int64)
    if len(y_test)!=16 or int(sum(y_test))!=3:
        raise RuntimeError('test必须为16家，风险3家')
    # Test is read only HERE, never in prepare/train or early stopping.
    for mode,(path,ckpt) in checkpoints.items():
        scaler=FinancialScaler.from_state(ckpt['scaler_state'])
        x=scaler.transform(test_raw)
        model=MultiHeadFinancialLSTM(**ckpt['model_config']).to(device)
        model.load_state_dict(ckpt['model_state'],strict=True)
        p,_=inference(model,x,y_test,int(cfg['training']['batch_size']),device)
        metrics=binary_metrics(y_test,p,float(ckpt['threshold']))
        entry={'mode':mode,'best_epoch':ckpt['best_epoch'],
               'checkpoint_sha256':sha256_file(path),
               'test_xlsx_sha256':sha256_file('data/splits/test.xlsx'),
               **metrics}
        records.append(entry)
        if not output.exists():
            output.mkdir(parents=True,exist_ok=False)
        json_save_exclusive(output/f'{mode}_metrics.json',entry)
        with (output/f'{mode}_company_predictions.csv').open('x',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f)
            w.writerow(['company_id','ground_truth_risk','risk_probability','predicted_risk','source'])
            for company_id,y,prob in zip(test_ids,y_test,p):
                w.writerow([int(company_id),int(y),float(prob),int(prob>=ckpt['threshold']),'real_test'])
        print(mode,'test AUROC=',metrics['auroc'],'PR-AUC=',metrics['pr_auc'],
              'F1=',metrics['f1_risk'],'balancedAcc=',metrics['balanced_accuracy'],flush=True)
    cols=['mode','n_companies','n_risk','accuracy','balanced_accuracy','precision_risk',
          'recall_risk','f1_risk','macro_f1','micro_f1','specificity','auroc',
          'pr_auc','ks','tn','fp','fn','tp','best_epoch']
    with (output/'test_comparison.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(records)
    json_save_exclusive(output/'evaluation_manifest.json',{
        'all_models_pretrained_and_selected_on_validation':True,
        'test_used_for_training_or_checkpoint_selection':False,
        'test_company_count':16,'risk_company_count':3,
        'test_xlsx_sha256':sha256_file('data/splits/test.xlsx'),
        'methods':list(MODES),
        'warning':'Only 3 risk companies in synthetic test; metrics are high variance. No real-world clinical/credit validity.'
    })
    print('最终测试评价完成：',output,flush=True)

if __name__=='__main__':
    main()
