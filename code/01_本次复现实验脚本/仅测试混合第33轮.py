"""冻结第33轮混合模型，仅回放测试数据，不训练或重新拟合归一化。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import 训练与统一测试 as base
from 训练与统一测试 import b, np, torch

def main():
    root=base.ROOT
    training=root/'训练结果'/'仓库算法_混合数据50轮_seed42'
    previous=root/'训练结果'/'仓库算法_新数据50轮_seed42'
    checkpoint=training/'最佳验证模型.pth'
    original=base.BASE/'冻结模型'/'仓库基线_50轮'/'LSTM离线修正模型.pth'
    pure=previous/'最佳验证模型.pth'
    out=root/'测试结果'/'混合模型_第33轮_冻结复测'
    if out.exists():
        raise FileExistsError(out)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['history'][-1]['epoch']==33, '检查点不是第33轮，停止'
    progress=json.loads((training/'训练进度.json').read_text(encoding='utf-8'))
    assert progress['selected_epoch']==33
    prior=json.loads((previous/'总评估结果.json').read_text(encoding='utf-8'))
    assert b.file_sha256(pure)==prior['new_model_sha256']
    oldtest=base.BASE/'数据集'/'吊载摆角测试数据集.csv'
    assert b.file_sha256(oldtest)==prior['repository_test_sha256']
    flights=base.accepted_flights()
    records=[(c,r) for c,r in flights if c['split']=='test']
    assert len(records)==4
    sources=[checkpoint,original,pure,oldtest,Path(__file__),Path(base.__file__)]+[Path(r['csv']) for _,r in records]
    hashes={str(p):b.file_sha256(p) for p in sources}
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(checkpoint,out/'混合LSTM_第33轮_冻结副本.pth')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models={'original_lstm':base.load_model(original,device),'new_lstm':base.load_model(checkpoint,device)}
    states={label:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for label,(model,_,_) in models.items()}
    print('确认检查点第33轮；仅冻结测试；device='+str(device),flush=True)
    perflight={};truths=[];errors={k:[] for k in ['ekf','original_lstm','new_lstm']}
    for cfg,record in records:
        name=cfg['flight_id']
        truth,err,info=base.evaluate(name,b.replay_csv(Path(record['csv'])),models,device,out)
        np.testing.assert_allclose(info['metrics']['ekf']['combined_rmse_deg'],prior['per_flight'][name]['metrics']['ekf']['combined_rmse_deg'],rtol=1e-10)
        info['new_only_reference']=prior['per_flight'][name]['metrics']['new_lstm']
        perflight[name]=info;truths.append(truth)
        for key in errors:
            errors[key].append(err[key])
    pooled=base.summary(np.concatenate(truths),{k:np.concatenate(v) for k,v in errors.items()})
    pooled['new_only_reference']=prior['new_test_pooled']['metrics']['new_lstm']
    _,_,oldinfo=base.evaluate('repository_test',b.replay_csv(oldtest),models,device,out)
    oldinfo['new_only_reference']=prior['repository_test']['metrics']['new_lstm']
    for label,(model,_,_) in models.items():
        assert all(torch.equal(v.detach().cpu(),states[label][k]) for k,v in model.state_dict().items())
    for path,digest in hashes.items():
        assert b.file_sha256(Path(path))==digest
    assert b.file_sha256(out/'混合LSTM_第33轮_冻结副本.pth')==hashes[str(checkpoint)]
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_epoch':33,
            'actual_completed_training_epochs':progress['epoch'],'training_performed_this_run':False,
            'weights_and_sources_unchanged':True,'source_hashes':hashes,'new_test_pooled':pooled,
            'per_flight':perflight,'repository_test':oldinfo,
            'metric_key_note':'new_lstm = mixed epoch33; new_only_reference = previous new-only epoch39.',
            'limitations':['Previously inspected test sets: comparison re-test, not blind validation.',
                          'One seed; original EKF and label issues preserved.',
                          'Mixed model selected using new-domain validation only.']}
    base.dump(out/'第33轮总评估结果.json',result)
    lines=['# 混合模型第33轮冻结测试','',
           '原混合训练计划50轮，按用户要求在完成35轮后中止；本次使用保存的第33轮最佳验证模型。',
           '本次没有训练、没有重新拟合归一化，模型和源文件哈希及内存权重核对通过。','',
           '| 数据 | EKF | 原模型+EKF | 仅新数据模型+EKF | 混合第33轮+EKF |','|---|---:|---:|---:|---:|']
    for name,info in [('四次新测试合并',pooled)]+list(perflight.items())+[('仓库原测试集',oldinfo)]:
        m=info['metrics'];vals=[m['ekf']['combined_rmse_deg'],m['original_lstm']['combined_rmse_deg'],info['new_only_reference']['combined_rmse_deg'],m['new_lstm']['combined_rmse_deg']]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(name,*vals))
    lines+=['','数值为综合摆角RMSE，单位度，越小越好。',
            '仅新数据模型使用此前相同文件、相同有效窗口的已完成测试指标作参考；本次实际重新推理原模型和混合第33轮模型。',
            '每个测试子目录保存预测CSV、指标JSON和对比PNG；图中的new LSTM指本次混合第33轮模型。',
            '已有测试集此前已查看，本次属于对照复测，不是全新盲测。单随机种子，验证集仅新工况；原EKF和标签问题未修正。']
    (out/'第33轮测试报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('第33轮冻结测试完成：'+str(out),flush=True)

if __name__=='__main__':
    main()
