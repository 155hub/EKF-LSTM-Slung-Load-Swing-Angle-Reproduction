"""用户明确授权的休眠中断恢复；复用原采集与训练逻辑，不改变算法。"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import 批量采集 as batch

ROOT=Path(__file__).resolve().parents[1]

def main():
    status=ROOT/'当前执行状态.json'
    previous=json.loads(status.read_text(encoding='utf-8'))
    if previous['status']!='failed' or previous['phase']!='formal_collection':
        raise RuntimeError('仅允许从已失败的采集阶段恢复')
    resultdir=ROOT/'训练结果'/'仓库算法_新数据50轮_seed42'
    if resultdir.exists():
        raise RuntimeError('已有训练目录，禁止重复训练或覆盖')
    for name in ['gzserver','px4','rosmaster']:
        if subprocess.run(['pgrep','-x',name],stdout=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('发现正在运行的仿真，拒绝恢复')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=ROOT/'恢复记录'
    archive.mkdir(exist_ok=True)
    shutil.copy2(status,archive/('恢复前状态_'+stamp+'.json'))
    protocol=batch.manifest()
    completed=[]
    pending=[]
    for cfg in protocol['flights']:
        if cfg['split']=='pilot':
            continue
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'采集记录').glob(cfg['flight_id']+'_attempt*.json'))]
        good=[r for r in records if r['accepted']]
        if len(good)>1:
            raise RuntimeError('重复合格记录')
        if good:
            r=good[0]
            if hashlib.sha256(Path(r['csv']).read_bytes()).hexdigest()!=r['sha256']:
                raise RuntimeError('已完成数据哈希变化')
            completed.append(cfg['flight_id'])
        else:
            start=max([r['attempt'] for r in records]+[0])+1
            while (ROOT/'原始数据'/cfg['split']/(cfg['flight_id']+'_attempt{:02d}.csv'.format(start))).exists():
                start+=1
            pending.append((cfg,start))
    state={'status':'running','phase':'formal_collection','pid':os.getpid(),'training_epochs':50,'seed':42,
           'started_utc':previous['started_utc'],'resumed_utc':datetime.now(timezone.utc).isoformat(),
           'recovery_reason':'User reports network disconnection and laptop sleep; resume authorized. Cause not independently verified.',
           'retained_flights':completed,'pending_flights':[c['flight_id'] for c,_ in pending]}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        status.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        with (ROOT/'运行日志'/'正式批量采集.log').open('a',buffering=1,encoding='utf-8') as log:
            with redirect_stdout(log):
                print('用户授权恢复 '+stamp+'；保留 '+str(len(completed))+' 次合格飞行。',flush=True)
                for cfg,start in pending:
                    for attempt in range(start,start+3):
                        print('恢复采集 {} attempt {}'.format(cfg['flight_id'],attempt),flush=True)
                        if batch.run_one(cfg,attempt)['accepted']:
                            break
                    else:
                        raise RuntimeError(cfg['flight_id']+'恢复后连续三次失败，停止等待处理')
                print('正式采集全部完成，进入训练。',flush=True)
        state.update(phase='training_and_testing',last_completed_phase='formal_collection')
        save()
        env=os.environ.copy()
        env['PYTHONUNBUFFERED']='1'
        env['PYTHONPATH']=str(ROOT.parent/'EKF-LSTM负载摆角估计_离线测试副本'/'Python依赖包_Ubuntu20.04_Py38')
        with (ROOT/'运行日志'/'正式训练与测试.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'源代码'/'训练与统一测试.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='complete',phase='complete',last_completed_phase='training_and_testing')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        save()
    print('恢复后的采集、训练与测试全部完成。',flush=True)

if __name__=='__main__':
    main()
