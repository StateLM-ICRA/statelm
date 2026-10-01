import argparse
from pathlib import Path
from .core import Inventory
from .data import prepare,read,write,make_test_n,fold_plan,split_cases
from .evaluation import evaluate_baselines,full_experiment

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description='StateLM: original-data experiments and separate Test N')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',default=str(ROOT/'data/source_review.json'));prep.add_argument('--output',default=str(ROOT/'data/original_cases.json'))
    for name in ('baselines','train'):
        q=sub.add_parser(name);q.add_argument('--cases',default=str(ROOT/'data/original_cases.json'));q.add_argument('--inventory',default=str(ROOT/'data/inventory.json'));q.add_argument('--output',required=True)
        if name=='train':
            q.add_argument('--config',default=str(ROOT/'config.json'));q.add_argument('--folds',type=int,nargs='+');q.add_argument('--seeds',type=int,nargs='+',default=[42]);q.add_argument('--test-n',action='store_true')
    demo=sub.add_parser('demo');demo.add_argument('message');demo.add_argument('--cart',choices=['lab','hospital'],default='lab')
    args=p.parse_args()
    if args.command=='prepare':
        r=prepare(args.source,args.output);print(r['counts']);print(r['exclusion_counts']);return
    if args.command=='demo':
        from .core import Agent
        a=Agent(Inventory(ROOT/'data/inventory.json'),condition='fsm')
        print(a.reply({'cart':args.cart,'history':[{'role':'participant','text':args.message}]}));return
    cases=read(args.cases)['cases'];inventory=Inventory(args.inventory)
    if args.command=='baselines':
        r=evaluate_baselines(cases,inventory,args.output)
        for name,tasks in r['conditions'].items():
            print(name,{t:{k:v for k,v in m.items() if k in ('n_cases','joint_item_drawer_correct','preference_agreement')} for t,m in tasks.items()})
    else:full_experiment(cases,inventory,args.output,read(args.config),args.folds,args.seeds,args.test_n)

if __name__=='__main__':main()
