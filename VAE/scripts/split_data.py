"""Split whole companies, never quarter rows; refuses to silently overwrite existing splits."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from VAE.src.dataset import read_financial,save_excel,stratified_company_split


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--overwrite',action='store_true',help='Recreate split files; invalidates existing checkpoints')
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    src=Path('data/raw/financial_80.xlsx')
    root=Path('data/splits')
    targets=[root/f'{s}.xlsx' for s in ('train','validation','test')]+[root/'split_manifest.json']
    if any(t.exists() for t in targets) and not args.overwrite:
        raise SystemExit('划分文件已存在，为防数据泄漏不能静默覆盖。查看 data/splits/ 或显式 --overwrite。')
    ids,x,labels=read_financial(src)
    split=stratified_company_split(ids,x,labels,args.seed)
    root.mkdir(parents=True,exist_ok=True)
    manifest={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
              'seed':args.seed,'split_unit':'company_id','splits':{}}
    for name,(si,sx,sl) in split.items():
        order=np.argsort(si)
        si,sx,sl=si[order],sx[order],sl[order]
        save_excel(root/f'{name}.xlsx',si,sx,sl)
        manifest['splits'][name]={'company_ids':si.tolist(),
           'count':int(len(si)), 'risk':int((sl=='有风险').sum()), 'normal':int((sl=='无风险').sum())}
        print(f'{name}: {len(si)}家（有风险 {(sl=="有风险").sum()} / 无风险 {(sl=="无风险").sum()}）')
    with (root/'split_manifest.json').open('w',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    print('划分已完成，split_manifest.json 保存每一集合企业ID与源文件哈希。')

if __name__=='__main__':main()
