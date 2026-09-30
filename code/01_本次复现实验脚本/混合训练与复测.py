"""原算法的混合训练对照。测试集仅复测，不参与归一化/训练/模型选择。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import 训练与统一测试 as base
from 训练与统一测试 import b, np, torch

ROOT=base.ROOT
OUT=ROOT/'训练结果'/'仓库算法_混合数据50轮_seed42'
PREVIOUS=ROOT/'训练结果'/'仓库算法_新数据50轮_seed42'

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    flights=base.accepted_flights()
    oldtrain=base.BASE/'数据集'/'吊载摆角训练数据集.csv'
    oldtest=base.BASE/'数据集'/'吊载摆角测试数据集.csv'
    original=base.BASE/'冻结模型'/'仓库基线_50轮'/'LSTM离线修正模型.pth'
    pure=PREVIOUS/'最佳验证模型.pth'
    prior=json.loads((PREVIOUS/'总评估结果.json').read_text(encoding='utf-8'))
    provenance=json.loads((PREVIOUS/'训练配置与数据来源.json').read_text(encoding='utf-8'))
    assert prior['repository_test_sha256']==b.file_sha256(oldtest)
    assert prior['new_model_sha256']==b.file_sha256(pure)
    assert {r['sha256'] for _,r in flights}=={r['sha256'] for r in provenance['flights']}
    paths=[oldtrain,oldtest,original,pure,Path(__file__),Path(base.__file__)]+[Path(p) for p in provenance['source_hashes']]
    hashes={str(p):b.file_sha256(p) for p in paths}
    OUT.mkdir(parents=True,exist_ok=False)
    settings={'seed':42,'epochs':50,'seq_len':400,'input_dim':13,'state_dim':7,'hidden_dim':64,'num_layers':2,
              'sampling_hz':50,'learning_rate':.001,'micro_batch':512,'effective_batch':2048,
              'mixture':'All windows of repository training CSV plus 10 new training flights, no oversampling.',
              'validation':'Same 4 new validation flights only; no old or new test data used in selection.',
              'selection':'minimum validation first-four-state sequence MSE',
              'normalization':'mixed training rows only','initialization':'from scratch, seed 42',
              'source_hashes':hashes,'flights':[r for _,r in flights],
              'limitations':['Repository Jacobian and label sign issues deliberately unchanged.',
                             'This is a re-test on previously inspected test sets, not a new blinded generalization claim.',
                             '50 epochs over a larger dataset imply more optimizer steps than new-only training.',
                             'Validation remains new-domain only; not balanced cross-domain model selection.']}
    base.dump(OUT/'训练配置与数据来源.json',settings)
    b.set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，停止')
    device=torch.device('cuda')
    train=[]; val=[]
    for cfg,record in flights:
        if cfg['split']=='test':
            continue
        print('回放 '+cfg['flight_id'],flush=True)
        replay=b.replay_csv(Path(record['csv']))
        (train if cfg['split']=='train' else val).append(replay)
    nx,ny=base.merge(train)
    print('回放仓库原训练集，保留时间重置边界',flush=True)
    old=b.replay_csv(oldtrain)
    tx=np.concatenate([nx,old.inputs]); ty=np.concatenate([ny,old.targets])
    vx,vy=base.merge(val)
    finite=np.isfinite(tx).all(axis=1)
    mean=tx[finite].mean(axis=0); std=tx[finite].std(axis=0)+1e-6
    train_ds=base.dataset(tx,ty,mean,std); val_ds=base.dataset(vx,vy,mean,std)
    counts={'new_train_windows':len(b.valid_window_starts(nx,400)),
            'old_train_windows':len(b.valid_window_starts(old.inputs,400)),
            'mixed_train_windows':len(train_ds),'validation_windows':len(val_ds),'old_replay':old.stats}
    assert counts['mixed_train_windows']==counts['new_train_windows']+counts['old_train_windows']
    base.dump(OUT/'窗口数量.json',counts)
    print(json.dumps(counts),flush=True)
    train_loader=b.make_loader(train_ds,512,True,0,device,42)
    val_loader=b.make_loader(val_ds,512,False,0,device,42)
    model=b.LSTMCorrectionModel(input_dim=13,state_dim=7,hidden_dim=64,num_layers=2).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001); loss=torch.nn.MSELoss()
    history=[]; best=float('inf'); selected=None
    cfg={k:settings[k] for k in ['seed','seq_len','input_dim','state_dim','hidden_dim','num_layers','sampling_hz']}
    for epoch in range(1,51):
        start=time.monotonic()
        tr=b.train_one_epoch(model,train_loader,optimizer,loss,device,2048)
        vl=b.validation_loss(model,val_loader,loss,device)
        assert np.isfinite([tr,vl]).all()
        history.append({'epoch':epoch,'train_loss':tr,'val_loss':vl,'seconds':time.monotonic()-start})
        ck={'config':cfg,'model_state_dict':model.state_dict(),'mean':mean,'std':std,'history':list(history)}
        if vl<best:
            best=vl; selected=epoch
            torch.save(ck,OUT/'最佳验证模型.pth')
        torch.save(ck,OUT/'最后一轮模型.pth')
        with (OUT/'训练进度.json').open('w',encoding='utf-8') as f:
            json.dump({'epoch':epoch,'epochs':50,'selected_epoch':selected,'history':history},f,ensure_ascii=False,indent=2)
        print('Epoch {}/50 train={:.8f} val={:.8f} time={:.1f}s best={}'.format(epoch,tr,vl,history[-1]['seconds'],selected),flush=True)
    fig,ax=b.plt.subplots(figsize=(8,4.5))
    ax.semilogy([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Mixed training')
    ax.semilogy([h['epoch'] for h in history],[h['val_loss'] for h in history],label='New-flight validation')
    ax.axvline(selected,color='grey',ls='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='MSE (first four states)',title='Repository LSTM | mixed training')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'训练损失曲线.png',dpi=160);b.plt.close(fig)
    models={'original_lstm':base.load_model(original,device),'new_lstm':base.load_model(OUT/'最佳验证模型.pth',device)}
    tests=OUT/'对照复测';tests.mkdir()
    truths=[];errors={k:[] for k in ['ekf','original_lstm','new_lstm']}; perflight={}
    for flight,record in flights:
        if flight['split']!='test':
            continue
        name=flight['flight_id']
        truth,err,info=base.evaluate(name,b.replay_csv(Path(record['csv'])),models,device,tests)
        np.testing.assert_allclose(info['metrics']['ekf']['combined_rmse_deg'],prior['per_flight'][name]['metrics']['ekf']['combined_rmse_deg'],rtol=1e-10)
        info['new_only_reference']=prior['per_flight'][name]['metrics']['new_lstm']
        perflight[name]=info;truths.append(truth)
        for key in errors:
            errors[key].append(err[key])
    pooled=base.summary(np.concatenate(truths),{k:np.concatenate(v) for k,v in errors.items()})
    pooled['new_only_reference']=prior['new_test_pooled']['metrics']['new_lstm']
    _,_,oldinfo=base.evaluate('repository_test',b.replay_csv(oldtest),models,device,tests)
    oldinfo['new_only_reference']=prior['repository_test']['metrics']['new_lstm']
    for path,digest in hashes.items():
        assert b.file_sha256(Path(path))==digest
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'selected_epoch':selected,
            'best_validation_loss':best,'window_counts':counts,'new_test_pooled':pooled,'per_flight':perflight,
            'repository_test':oldinfo,'source_hashes_unchanged':True,
            'mixed_model_sha256':b.file_sha256(OUT/'最佳验证模型.pth'),
            'metric_key_note':'new_lstm means the mixed-trained model in this directory; new_only_reference is the prior model trained only on new data.',
            'limitations':settings['limitations']}
    base.dump(OUT/'总评估结果.json',result)
    lines=['# 混合训练对照复测','',
           '新模型从头训练50轮，依据相同四次新验证飞行选择第{}轮。'.format(selected),
           '原仓库训练集与10次新训练飞行按全部有效窗口混合，未使用测试数据训练或归一化。','',
           '| 数据 | EKF | 原模型+EKF | 仅新数据模型+EKF | 混合模型+EKF |',
           '|---|---:|---:|---:|---:|']
    for name,info in [('四次新测试合并',pooled)]+list(perflight.items())+[('仓库原测试',oldinfo)]:
        m=info['metrics']
        values=[m['ekf']['combined_rmse_deg'],m['original_lstm']['combined_rmse_deg'],
                info['new_only_reference']['combined_rmse_deg'],m['new_lstm']['combined_rmse_deg']]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(name,*values))
    lines+=['','单位为度，综合RMSE越小越好。','',
            '已有测试集已在此前查看，本次是对照复测，不是全新盲测。原模型与仅新数据模型未改动。',
            '验证集仅覆盖新工况，50轮混合训练有更多优化步数；不能把变化全部归因于数据比例。',
            '只运行种子42，EKF雅可比和标签已知问题按用户要求保留。',
            'JSON中的new_lstm及图中的new LSTM均指本次混合模型。']
    (OUT/'混合训练对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('混合训练与对照复测全部完成。',flush=True)

if __name__=='__main__':
    main()
