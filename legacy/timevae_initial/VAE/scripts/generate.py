"""Sample from one conditional VAE by class, export separate XLSX, then evaluate automatically."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch,yaml
from VAE.src.dataset import FinancialScaler,read_financial,save_excel
from VAE.src.finance_rules import repair_financial
from VAE.src.metrics import metric_report
from VAE.src.schema import NAMES
from VAE.src.timevae import TimeVAE
from VAE.scripts.train import seed_all,device_from
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description='--class risk/normal/all and --number per selected class')
    p.add_argument('--config',default='config/vae.yaml')
    p.add_argument('--class',dest='target',choices=['risk','normal','all'],default='all')
    p.add_argument('--number',type=int,default=None,help='Number per selected class; for all, number EACH')
    p.add_argument('--risk-number',type=int,default=None)
    p.add_argument('--normal-number',type=int,default=None)
    p.add_argument('--device',default=None)
    p.add_argument('--temperature',type=float,default=None)
    p.add_argument('--output',default=None,help='Unique output directory')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    num_default=args.number if args.number is not None else cfg['generation']['default_number_per_class']
    if num_default <= 0:raise SystemExit('生成数量必须大于0')
    if args.target!='all' and (args.risk_number is not None or args.normal_number is not None):
        raise SystemExit('--risk-number/--normal-number仅在--class all时指定')
    selected={'risk':args.risk_number if args.risk_number is not None else num_default,
              'normal':args.normal_number if args.normal_number is not None else num_default}
    if any(n<1 for n in selected.values()):raise SystemExit('生成数量必须大于0')
    temp=args.temperature if args.temperature is not None else cfg['generation']['temperature']
    if not 0.1<=temp<=3.:raise SystemExit('temperature必须在[0.1,3]范围')
    seed_all(cfg['generation']['random_seed']);torch.set_num_threads(min(torch.get_num_threads(),4))
    device=device_from(args.device or cfg['training']['device'])
    ckpt=ROOT/'VAE/outputs/checkpoints/timevae_conditional_best.pt'
    if not ckpt.exists():raise SystemExit(f'尚无训练模型：{ckpt}，先运行train.py')
    saved=torch.load(ckpt,map_location='cpu',weights_only=False) # ONLY trusted, self-created checkpoint
    if saved.get('feature_names')!=NAMES or saved.get('label_map')!={'无风险':0,'有风险':1}:
        raise RuntimeError('checkpoint的财务指标或条件编码不匹配')
    if saved.get('split_manifest_sha256') != hashlib.sha256((ROOT/'data/splits/split_manifest.json').read_bytes()).hexdigest():
        raise RuntimeError('checkpoint与当前划分不匹配；不能生成。')
    for file_key,filename in [('train_xlsx_sha256','train.xlsx'),('validation_xlsx_sha256','validation.xlsx')]:
        present=hashlib.sha256((ROOT/'data/splits'/filename).read_bytes()).hexdigest()
        if saved.get(file_key)!=present:
            raise RuntimeError(f'{filename}内容与checkpoint训练时不匹配，拒绝生成。')
    model=TimeVAE(**saved['model_params']).to(device)
    model.load_state_dict(saved['model_state'],strict=True);model.eval()
    scaler=FinancialScaler.from_state(saved['scaler_state'])
    out=ROOT/(args.output or f'VAE/outputs/generated/run_seed{cfg["generation"]["random_seed"]}')
    evaluation=ROOT/'VAE/outputs/evaluation'/out.name
    if (out.exists() and any(out.iterdir())) or (evaluation.exists() and any(evaluation.iterdir())):
        raise SystemExit(f'生成或评价输出已存在：{out} / {evaluation}；请改用新--output')
    out.mkdir(parents=True,exist_ok=True);evaluation.mkdir(parents=True,exist_ok=True)
    ids,x,y=read_financial(ROOT/'data/splits/train.xlsx')
    vi,vx,vy=read_financial(ROOT/'data/splits/validation.xlsx')
    keys=['risk','normal'] if args.target=='all' else [args.target]
    outputs=[];summary={}
    for key in keys:
        label='有风险' if key=='risk' else '无风险';category=1 if key=='risk' else 0
        count=selected[key]
        standardized=model.sample(count,device,condition=category,temperature=temp)
        raw=scaler.inverse_transform(standardized)
        clean=repair_financial(raw)
        new_ids=np.arange((100000 if key=='risk' else 200000)+1,
                          (100000 if key=='risk' else 200000)+count+1)
        lab=np.repeat(label,count)
        filepath=out/f'generated_{key}_{count}.xlsx'
        save_excel(filepath,new_ids,clean,lab)
        print(f'已导出：{filepath}，共 {count} 家企业',flush=True)
        outputs.append((new_ids,clean,lab))
        report=metric_report(x[y==label],clean,vx[vy==label],scaler,evaluation,key,raw_generated=raw)
        summary[key]=report
        print(f'{key}: COS={report["cosine_nn_train_mean"]:.4f}; Coverage={report["coverage_train"]:.4f}; '
              f'MMD²={report["rbf_mmd2"]:.4f}; 数据修正率={report["postprocessing_changed_point_fraction"]:.3f}',flush=True)
    if len(outputs)==2:
        save_excel(out/'generated_all.xlsx',
            np.concatenate([a for a,_,_ in outputs]),np.concatenate([b for _,b,_ in outputs]),
            np.concatenate([c for _,_,c in outputs]))
    (evaluation/'summary_all.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('自动评价完成：',evaluation)
    print('测试集未用于模型训练、归一化、生成或评价。')

if __name__=='__main__':main()
