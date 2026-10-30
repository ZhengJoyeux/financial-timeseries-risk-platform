"""Regenerate the synthetic 80-firm dataset. WARNING: overwrite only with --overwrite."""
import argparse
from pathlib import Path
from VAE.src.synthetic_data import build_simulated_companies
from VAE.src.dataset import save_excel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    path=Path('data/raw/financial_80.xlsx')
    if path.exists() and not args.overwrite:
        raise SystemExit(f'文件已存在: {path}; 避免覆盖数据。确实要重新生成，请显式添加 --overwrite。')
    ids,x,labels=build_simulated_companies(args.seed)
    save_excel(path,ids,x,labels)
    print(f'已创建: {path}; 80家、16家有风险、64家无风险、20季度、15变量。')

if __name__=='__main__':main()
