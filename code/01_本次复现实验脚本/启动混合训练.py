from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
status=ROOT/'混合训练执行状态.json'
if status.exists():
    raise FileExistsError('已有混合训练状态，禁止重复启动')
state={'status':'running','phase':'mixed_training_and_retest','pid':os.getpid(),
       'started_utc':datetime.now(timezone.utc).isoformat(),'epochs':50,'seed':42}
def save():
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    status.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
save()
try:
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=str(ROOT.parent/'EKF-LSTM负载摆角估计_离线测试副本'/'Python依赖包_Ubuntu20.04_Py38')
    with (ROOT/'运行日志'/'混合训练与复测.log').open('x') as log:
        subprocess.run([sys.executable,str(ROOT/'源代码'/'混合训练与复测.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state.update(status='complete',phase='complete')
except BaseException as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    save()
