"""WSL20 顺序执行独立 SITL 飞行，保留失败记录，不接触物理飞行器。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
from 轨迹协议 import trajectory

ROOT=Path(__file__).resolve().parents[1]
SIM=ROOT.parent/'独立仿真实验'

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def manifest():
    path=ROOT/'飞行协议.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    flights=[]
    for group,count,offset in [('train',10,100),('validation',4,200),('test',4,300),('pilot',1,900)]:
        for i in range(count):
            seed=20260914+offset+i
            r=random.Random(seed)
            kinds=['single_axis','lissajous','ellipse','burst']
            r.shuffle(kinds)
            if group=='test' and i>=2:
                kinds[2]='chirp'
            stages=[]
            for k,a in zip(kinds,[.4,.65,.95,1.15]):
                stages.append({'kind':k,'amplitude_m':a*r.uniform(.93,1.07),
                               'omega_rad_s':r.uniform(1.65,1.9),'phase_rad':r.uniform(-math.pi,math.pi),
                               'rotation_rad':r.uniform(-math.pi,math.pi),'ramp_s':r.uniform(3.5,5.)})
            cfg={'flight_id':'{}_{:02d}'.format(group,i+1),'split':group,'seed':seed,
                 'duration_s':180,'stages':stages,'test_kind':'held_out_chirp' if group=='test' and i>=2 else 'same_family_new_parameters'}
            for t in [j*.02 for j in range(9001)]:
                xy=trajectory(t,cfg)
                assert all(math.isfinite(v) and abs(v)<3 for v in xy)
            flights.append(cfg)
    doc={'protocol_version':1,'created_before_collection':True,'flights':flights,
         'quality_rule':'complete + disarmed + >=99% rows + positive dt <=0.060001 s + finite + no zero thrust',
         'selection':'Only data/control quality causes rejection. Never select by LSTM performance.'}
    dump(path,doc)
    for cfg in flights:
        dump(ROOT/'轨迹配置'/(cfg['flight_id']+'.json'),cfg)
    return doc

def quality(path):
    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with path.open() as f:
        rows=list(csv.DictReader(f))
    ts=[float(r['timestamp']) for r in rows]
    dt=[b-a for a,b in zip(ts,ts[1:])]
    finite=all(math.isfinite(float(v)) for r in rows for v in r.values())
    zero=sum(float(r['cmd_thrust'])==0 for r in rows)
    accepted=(meta['status']=='complete' and meta.get('landed_disarmed',False) and len(rows)>=8910
              and bool(dt) and min(dt)>0 and max(dt)<=.060001 and finite and zero==0)
    peaks={a:max(abs(math.degrees(float(r['gt_'+a]))) for r in rows) for a in ['xi','zeta']} if rows else {}
    return {'accepted':accepted,'rows':len(rows),'finite':finite,'zero_thrust':zero,
            'dt_min':min(dt) if dt else None,'dt_max':max(dt) if dt else None,'peak_abs_deg':peaks,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def run_one(cfg,attempt):
    name=cfg['flight_id']+'_attempt{:02d}'.format(attempt)
    folder=ROOT/'原始数据'/cfg['split']
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/(name+'.csv')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    logs=ROOT/'运行日志'
    logs.mkdir(exist_ok=True)
    # Refuse to stop or share an existing simulator owned by another run.
    for process in ['gzserver','px4','rosmaster']:
        if subprocess.run(['pgrep','-x',process],stdout=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('已有进程，禁止启动冲突仿真: '+process)
    before=set(Path('/home/ubuntu/.ros/log').rglob('*.ulg'))
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.pop('TEST_SDF',None)
    launch=None
    code=None
    with (logs/(name+'_仿真.log')).open('x') as lf, (logs/(name+'_采集.log')).open('x') as cf:
        try:
            launch=subprocess.Popen(['bash',str(SIM/'启动独立仿真.sh')],env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            time.sleep(12)
            if launch.poll() is not None:
                raise RuntimeError('仿真启动提前退出')
            command='source "$1"; exec python3 "$2" --config "$3" --output "$4"'
            args=['bash','-c',command,'collector',str(SIM/'仿真环境.sh'),str(ROOT/'源代码'/'采集多轨迹飞行.py'),
                  str(ROOT/'轨迹配置'/(cfg['flight_id']+'.json')),str(output)]
            code=subprocess.run(args,env=env,stdout=cf,stderr=subprocess.STDOUT,timeout=800).returncode
        finally:
            if launch is not None and launch.poll() is None:
                os.killpg(launch.pid,signal.SIGINT)
                try:
                    launch.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(launch.pid,signal.SIGTERM)
                    launch.wait(timeout=15)
    raw_logs=[]
    for log in sorted(set(Path('/home/ubuntu/.ros/log').rglob('*.ulg'))-before):
        target=folder/(name+'_飞控日志_'+str(len(raw_logs)+1)+'.ulg')
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(log,target)
        raw_logs.append({'original':str(log),'copy':str(target)})
    result={'flight_id':cfg['flight_id'],'split':cfg['split'],'attempt':attempt,'csv':str(output),
            'collector_exit_code':code,'ulog_copies':raw_logs,'accepted':False}
    if code==0:
        result.update(quality(output))
    dump(ROOT/'采集记录'/(name+'.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    doc=manifest()
    if args.prepare_only:
        print('协议已固定，18次正式飞行和1次试飞。',flush=True)
        return
    for cfg in doc['flights']:
        if (cfg['split']=='pilot') != args.pilot:
            continue
        previous=[]
        for file in sorted((ROOT/'采集记录').glob(cfg['flight_id']+'_attempt*.json')):
            previous.append(json.loads(file.read_text(encoding='utf-8')))
        if any(p['accepted'] for p in previous):
            continue
        for attempt in range(len(previous)+1,4):
            print('开始 {} attempt {}'.format(cfg['flight_id'],attempt),flush=True)
            if run_one(cfg,attempt)['accepted']:
                break
        else:
            raise RuntimeError(cfg['flight_id']+'连续三次未通过采集质量要求，请检查记录')
    print('本批采集完成。',flush=True)

if __name__=='__main__':
    main()
