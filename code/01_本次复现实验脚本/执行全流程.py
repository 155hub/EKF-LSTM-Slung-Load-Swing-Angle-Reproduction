"""执行正式采集，然后训练和测试；任何子流程失败即停止并记录。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    status=ROOT/'当前执行状态.json'
    if status.exists():
        raise FileExistsError('已有执行状态，禁止重复启动全流程')
    state={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
           'status':'running','phase':'formal_collection','training_epochs':50,'seed':42}
    def save():
        with status.open('w',encoding='utf-8') as f:
            json.dump(state,f,ensure_ascii=False,indent=2)
    save()
    logdir=ROOT/'运行日志'
    logdir.mkdir(exist_ok=True)
    try:
        for phase,script,filename in [('formal_collection','批量采集.py','正式批量采集.log'),
                                       ('training_and_testing','训练与统一测试.py','正式训练与测试.log')]:
            state['phase']=phase
            save()
            env=os.environ.copy()
            env['PYTHONUNBUFFERED']='1'
            env.pop('PYTHONPATH',None)
            if phase=='training_and_testing':
                env['PYTHONPATH']=str(ROOT.parent/'EKF-LSTM负载摆角估计_离线测试副本'/'Python依赖包_Ubuntu20.04_Py38')
            with (logdir/filename).open('x') as log:
                print('开始阶段: '+phase,flush=True)
                subprocess.run([sys.executable,str(ROOT/'源代码'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['last_completed_phase']=phase
            save()
        state.update(status='complete',phase='complete')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        save()
    print('全部完成，见训练结果。',flush=True)

if __name__=='__main__':
    main()
