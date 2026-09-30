"""仓库原算法，新独立飞行训练/验证/测试。只在本轮目录写结果。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'EKF-LSTM负载摆角估计_离线测试副本'
sys.path.insert(0,str(BASE/'源代码'))
import 离线效果验证 as b
import numpy as np
import pandas as pd
import torch


def dump(path,data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def accepted_flights():
    protocol=json.loads((ROOT/'飞行协议.json').read_text(encoding='utf-8'))
    result=[]
    for cfg in protocol['flights']:
        if cfg['split']=='pilot':
            continue
        candidates=[]
        for file in sorted((ROOT/'采集记录').glob(cfg['flight_id']+'_attempt*.json')):
            record=json.loads(file.read_text(encoding='utf-8'))
            if record['accepted']:
                candidates.append(record)
        if len(candidates)!=1:
            raise RuntimeError('每次正式飞行必须恰好有一份通过质量门槛的数据: '+cfg['flight_id'])
        record=candidates[0]
        path=Path(record['csv'])
        if b.file_sha256(path)!=record['sha256']:
            raise RuntimeError('采集后数据被修改: '+str(path))
        result.append((cfg,record))
    assert len({r['sha256'] for _,r in result})==18
    return result


def merge(replays):
    return np.concatenate([r.inputs for r in replays]),np.concatenate([r.targets for r in replays])


def dataset(inputs,targets,mean,std):
    normalized=(inputs-mean)/std
    starts=b.valid_window_starts(normalized,400)
    assert len(starts)>0
    return b.WindowDataset(normalized,targets,starts,400)


def metric(error):
    return {'xi_rmse_deg':float(b.rmse(error[:,0])), 'zeta_rmse_deg':float(b.rmse(error[:,1])),
            'combined_rmse_deg':float(b.rmse(error)),
            'mae_deg':np.mean(np.abs(error),axis=0).tolist(),
            'max_abs_deg':np.max(np.abs(error),axis=0).tolist()}


def summary(truth,errors):
    values={key:metric(err) for key,err in errors.items()}
    denominator=values['ekf']['combined_rmse_deg']
    for key in ['original_lstm','new_lstm']:
        values[key]['improvement_percent']=100*(1-values[key]['combined_rmse_deg']/denominator) if denominator>0 else None
    bins=[]
    size=np.max(np.abs(truth),axis=1)
    for lo,hi in [(0,5),(5,15),(15,25),(25,180)]:
        mask=(size>=lo)&(size<hi)
        bins.append({'lower_deg':lo,'upper_deg':hi,'windows':int(mask.sum()),
                     'metrics':{key:metric(err[mask]) for key,err in errors.items()} if mask.any() else None})
    return {'windows':len(truth),'metrics':values,'angle_bins':bins}


def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ck['config']
    assert cfg['seq_len']==400 and cfg['input_dim']==13 and cfg['state_dim']==7
    model=b.LSTMCorrectionModel(input_dim=13,state_dim=7,hidden_dim=64,num_layers=2).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval().requires_grad_(False)
    return model,np.asarray(ck['mean']),np.asarray(ck['std'])


def evaluate(name,replay,models,device,out):
    folder=out/name
    folder.mkdir()
    predictions={}
    reference_endpoints=None
    for label,(model,mean,std) in models.items():
        ds=dataset(replay.inputs,replay.targets,mean,std)
        loader=b.make_loader(ds,512,False,0,device,42)
        with torch.inference_mode():
            values,endpoints=b.evaluate_endpoints(model,loader,device)
        assert np.isfinite(values).all()
        if reference_endpoints is None:
            reference_endpoints=endpoints
        else:
            np.testing.assert_array_equal(endpoints,reference_endpoints)
        predictions[label]=values[:,:2]
    idx=reference_endpoints
    truth=np.rad2deg(replay.inputs[idx,:2]+replay.targets[idx,:2])
    ekf=np.rad2deg(replay.inputs[idx,:2])
    errors={'ekf':truth-ekf}
    for label,values in predictions.items():
        errors[label]=truth-(ekf+np.rad2deg(values))
    info=summary(truth,errors)
    info['replay']=replay.stats
    data={'time_s':replay.timestamps[idx]}
    for j,axis in enumerate(['xi','zeta']):
        data['true_'+axis+'_deg']=truth[:,j]
        for key,err in errors.items():
            data[key+'_'+axis+'_deg']=truth[:,j]-err[:,j]
            data[key+'_error_'+axis+'_deg']=err[:,j]
    pd.DataFrame(data).to_csv(folder/'三方逐窗口预测.csv',index=False)
    dump(folder/'评估指标.json',info)
    fig,axes=b.plt.subplots(2,1,figsize=(13,7),sharex=True)
    times=data['time_s']-data['time_s'][0]
    for j,ax in enumerate(axes):
        ax.plot(times,truth[:,j],label='Ground truth',lw=1.2,color='black')
        for key,label in [('ekf','EKF'),('original_lstm','EKF + original LSTM'),('new_lstm','EKF + new LSTM')]:
            ax.plot(times,truth[:,j]-errors[key][:,j],label=label,lw=.8,alpha=.85)
        ax.set_ylabel(['Xi (deg)','Zeta (deg)'][j])
        ax.grid(alpha=.25)
    axes[0].legend(ncol=2)
    axes[1].set_xlabel('Time from first evaluated endpoint (s)')
    fig.suptitle(name+' | frozen evaluation')
    fig.tight_layout()
    fig.savefig(folder/'三方摆角对比.png',dpi=160)
    b.plt.close(fig)
    print(name,json.dumps(info['metrics']),flush=True)
    return truth,errors,info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        x=np.vstack([np.zeros((410,13)),np.full((1,13),np.nan),np.ones((420,13)),np.full((1,13),np.nan)])
        starts=b.valid_window_starts(x,400)
        assert len(starts)==11+21
        assert all(np.isfinite(x[s:s+400]).all() for s in starts)
        assert b.SlungLoadEKF().L==2
        print('窗口边界测试通过；CUDA=',torch.cuda.is_available(),flush=True)
        return
    flights=accepted_flights()
    out=ROOT/'训练结果'/('仓库算法_新数据{}轮_seed{}'.format(args.epochs,args.seed))
    out.mkdir(parents=True,exist_ok=False)
    b.set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('预期 CUDA 不可用，停止避免意外长时间 CPU 训练')
    frozen=BASE/'冻结模型'/'仓库基线_50轮'/'LSTM离线修正模型.pth'
    frozen_hash=b.file_sha256(frozen)
    hashes={str(BASE/'源代码'/name):b.file_sha256(BASE/'源代码'/name)
            for name in ['吊载扩展卡尔曼滤波.py','离线效果验证.py','LSTM修正模型.py']}
    settings={'seed':args.seed,'epochs':args.epochs,'seq_len':400,'input_dim':13,'state_dim':7,
              'hidden_dim':64,'num_layers':2,'sampling_hz':50,'lr':.001,'micro_batch':512,
              'effective_batch':2048,'selection':'minimum validation MSE on first four states across all timesteps',
              'normalization':'training flights only','algorithm':'unmodified repository EKF and target signs',
              'original_model_sha256':frozen_hash,'source_hashes':hashes,'flights':[r for _,r in flights],
              'protocol_sha256':b.file_sha256(ROOT/'飞行协议.json'),
              'driver_sha256':b.file_sha256(Path(__file__)),
              'python':sys.version,'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    dump(out/'训练配置与数据来源.json',settings)
    cache=ROOT/'回放缓存'
    cache.mkdir(exist_ok=True)
    replays={}
    # Test replay deferred until model selection is complete.
    for cfg,record in flights:
        if cfg['split']=='test':
            continue
        print('回放',cfg['flight_id'],flush=True)
        replay=b.replay_csv(Path(record['csv']))
        replays[cfg['flight_id']]=replay
        cachefile=cache/(cfg['flight_id']+'.npz')
        if cachefile.exists():
            raise FileExistsError(cachefile)
        np.savez_compressed(cachefile,inputs=replay.inputs,targets=replay.targets,timestamps=replay.timestamps)
        dump(cache/(cfg['flight_id']+'.json'),{'source_sha256':record['sha256'],'replay':replay.stats,'source':record['csv']})
    train=[replays[c['flight_id']] for c,_ in flights if c['split']=='train']
    val=[replays[c['flight_id']] for c,_ in flights if c['split']=='validation']
    tx,ty=merge(train)
    vx,vy=merge(val)
    finite=np.isfinite(tx).all(axis=1)
    mean=tx[finite].mean(axis=0)
    std=tx[finite].std(axis=0)+1e-6
    train_ds=dataset(tx,ty,mean,std)
    val_ds=dataset(vx,vy,mean,std)
    print('windows train={} validation={} device={}'.format(len(train_ds),len(val_ds),device),flush=True)
    train_loader=b.make_loader(train_ds,512,True,0,device,args.seed)
    val_loader=b.make_loader(val_ds,512,False,0,device,args.seed)
    model=b.LSTMCorrectionModel(input_dim=13,state_dim=7,hidden_dim=64,num_layers=2).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss=torch.nn.MSELoss()
    history=[]
    best=float('inf')
    selected_epoch=None
    config={k:settings[k] for k in ['seed','seq_len','input_dim','state_dim','hidden_dim','num_layers','sampling_hz']}
    for epoch in range(1,args.epochs+1):
        start=time.monotonic()
        tr=b.train_one_epoch(model,train_loader,optimizer,loss,device,2048)
        vl=b.validation_loss(model,val_loader,loss,device)
        if not np.isfinite([tr,vl]).all():
            raise RuntimeError('损失非有限，停止')
        history.append({'epoch':epoch,'train_loss':tr,'val_loss':vl,'seconds':time.monotonic()-start})
        ck={'config':config,'model_state_dict':model.state_dict(),'mean':mean,'std':std,'history':list(history)}
        if vl<best:
            best=vl
            selected_epoch=epoch
            torch.save(ck,out/'最佳验证模型.pth')
        torch.save(ck,out/'最后一轮模型.pth')
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'history':history,'best_val':best,'selected_epoch':selected_epoch,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'训练中断恢复信息.pth')
        with (out/'训练进度.json').open('w',encoding='utf-8') as f:
            json.dump({'epoch':epoch,'epochs':args.epochs,'selected_epoch':selected_epoch,'history':history},f,ensure_ascii=False,indent=2)
        print('Epoch {}/{} train={:.8f} val={:.8f} time={:.1f}s best={}'.format(epoch,args.epochs,tr,vl,history[-1]['seconds'],selected_epoch),flush=True)
    fig,ax=b.plt.subplots(figsize=(8,4.5))
    ax.semilogy([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Train flights')
    ax.semilogy([h['epoch'] for h in history],[h['val_loss'] for h in history],label='Validation flights')
    ax.axvline(selected_epoch,color='grey',ls='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='MSE (first four states)',title='New flight dataset | repository LSTM')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'训练损失曲线.png',dpi=160); b.plt.close(fig)
    models={'original_lstm':load_model(frozen,device),'new_lstm':load_model(out/'最佳验证模型.pth',device)}
    tests=out/'独立测试'
    tests.mkdir()
    truths=[]; errors={k:[] for k in ['ekf','original_lstm','new_lstm']}; perflight={}
    for cfg,record in flights:
        if cfg['split']!='test':
            continue
        replay=b.replay_csv(Path(record['csv']))
        truth,err,info=evaluate(cfg['flight_id'],replay,models,device,tests)
        truths.append(truth)
        for key in errors:
            errors[key].append(err[key])
        perflight[cfg['flight_id']]=info
    pooled=summary(np.concatenate(truths),{k:np.concatenate(v) for k,v in errors.items()})
    # Secondary repository test: never used to choose a new model.
    oldcsv=BASE/'数据集'/'吊载摆角测试数据集.csv'
    _,_,oldinfo=evaluate('repository_test',b.replay_csv(oldcsv),models,device,tests)
    for path,digest in hashes.items():
        assert b.file_sha256(Path(path))==digest
    assert b.file_sha256(frozen)==frozen_hash
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'selected_epoch':selected_epoch,
            'best_validation_loss':best,'train_windows':len(train_ds),'validation_windows':len(val_ds),
            'new_test_pooled':pooled,'per_flight':perflight,'repository_test':oldinfo,
            'repository_test_sha256':b.file_sha256(oldcsv),'original_model_unchanged':True,
            'new_model_sha256':b.file_sha256(out/'最佳验证模型.pth'),
            'known_issues':'Original Jacobians and target signs preserved at user request; no physical correctness claim.'}
    dump(out/'总评估结果.json',result)
    lines=['# 新数据训练与独立测试结果','',
           '原仓库 EKF 与标签定义未修正；新数据按完整飞行划分。',
           '模型选择依据验证集损失，选中第 {} 轮。'.format(selected_epoch),'',
           '| 数据 | EKF RMSE (deg) | 原 LSTM + EKF | 新 LSTM + EKF |','|---|---:|---:|---:|']
    for name,info in [('新独立测试集合并',pooled)]+list(perflight.items())+[('仓库原测试集（补充）',oldinfo)]:
        m=info['metrics']
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(name,*[m[k]['combined_rmse_deg'] for k in ['ekf','original_lstm','new_lstm']]))
    lines+=['','整体指标按有效窗口合并计算，各飞行另列；窗口高度重叠，不能当作独立重复实验。',
            '仓库测试集只作补充对照，未用于模型选择。大摆角区间与分轴指标见总评估结果.json。',
            '本次只运行一个训练随机种子，尚未检验不同初始化的稳定性。',
            '原模型和源文件哈希复核通过，没有覆盖旧基线。']
    with (out/'训练测试结果说明.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('全部训练与测试完成。',flush=True)


if __name__=='__main__':
    main()
