"""确定性位置指令；不修改仓库 EKF 或 LSTM。"""
import math

def trajectory(t, cfg):
    if not 12 <= t < 156:
        return 0., 0.
    stage = min(3, int((t-12)//36))
    s = t-12-36*stage
    p = cfg['stages'][stage]
    ramp = max(0., min(1., s/p['ramp_s'], (36-s)/p['ramp_s']))
    env = .5-.5*math.cos(math.pi*ramp)
    a,w,phase = p['amplitude_m'],p['omega_rad_s'],p['phase_rad']
    kind = p['kind']
    if kind == 'lissajous':
        x,y = a*math.sin(w*s+phase), .8*a*math.sin(.89*w*s-phase)
    elif kind == 'ellipse':
        x,y = a*math.sin(w*s+phase), .75*a*math.cos(w*s+phase)
    elif kind == 'single_axis':
        x,y = a*math.sin(w*s+phase), 0.
    elif kind == 'burst':
        gate = math.sin(math.pi*min(s%12,6)/6)**2 if s%12 < 6 else 0.
        x,y = gate*a*math.sin(w*s+phase), gate*.8*a*math.sin(.9*w*s)
    elif kind == 'chirp':
        theta=w*s+.5*.015*s*s
        x,y=a*math.sin(theta+phase), .7*a*math.sin(.83*theta-phase)
    else:
        raise ValueError(kind)
    c,sn=math.cos(p['rotation_rad']),math.sin(p['rotation_rad'])
    return env*(c*x-sn*y),env*(sn*x+c*y)
