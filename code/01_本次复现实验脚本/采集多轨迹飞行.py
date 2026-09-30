"""纯 SITL 分级摆角激励；不修改飞控检查或冻结网络。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import DeleteModel
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import SetMode, CommandBool, MessageInterval
from 轨迹协议 import trajectory

HEADER = ['timestamp','imu_ax','imu_ay','imu_az','imu_wx','imu_wy','imu_wz',
          'drone_px','drone_py','drone_pz','drone_vx','drone_vy','drone_vz',
          'drone_qw','drone_qx','drone_qy','drone_qz','cmd_thrust',
          'gt_xi','gt_zeta','gt_xi_dot','gt_zeta_dot','gt_load_px','gt_load_py','gt_load_pz',
          'gt_drone_px','gt_drone_py','gt_drone_pz','imu_stamp','pose_stamp','thrust_stamp','max_receive_age_s',
          'gt_drone_qw','gt_drone_qx','gt_drone_qy','gt_drone_qz','gt_drone_vx','gt_drone_vy','gt_drone_vz']


class Flight:
    def __init__(self, config):
        self.config = config
        self.msg = {}
        self.stamps = {}
        self.lock = threading.Lock()
        self.rows = 0
        self.skips = 0
        self.prev = None
        self.pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=5)
        for name, topic, cls in [('state','/mavros/state',State),('pose','/mavros/local_position/pose',PoseStamped),
                                  ('vel','/mavros/local_position/velocity_local',TwistStamped),
                                  ('imu','/mavros/imu/data',Imu),('cmd','/mavros/setpoint_raw/target_attitude',AttitudeTarget),
                                  ('links','/gazebo/link_states',LinkStates)]:
            rospy.Subscriber(topic, cls, self.receive, callback_args=name, queue_size=1)
        self.mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arm = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

    def receive(self, msg, name):
        with self.lock:
            self.msg[name] = msg
            self.stamps[name] = rospy.get_time()

    def setpoint(self, x, y, z):
        p = PoseStamped()
        p.header.stamp = rospy.Time.now()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = x,y,z
        p.pose.orientation.w = 1
        self.pub.publish(p)

    def sample(self, writer):
        with self.lock:
            d = dict(self.msg)
            stamps = dict(self.stamps)
        t = rospy.get_time()
        if not all(k in d for k in ['imu','pose','vel','cmd','links','state']):
            self.skips += 1
            return
        age = max(t-stamps[k] for k in ['imu','pose','vel','cmd','links'])
        if age > .1 or not d['state'].armed or d['state'].mode != 'OFFBOARD':
            self.skips += 1
            return
        links = d['links']
        dp = links.pose[links.name.index('iris::base_link')].position
        gq = links.pose[links.name.index('iris::base_link')].orientation
        gv = links.twist[links.name.index('iris::base_link')].linear
        lp = links.pose[links.name.index('iris::payload')].position
        dx,dy,dz = lp.x-dp.x, lp.y-dp.y, lp.z-dp.z
        xi,zeta = math.atan2(dx,-dz),math.atan2(dy,-dz)
        if self.prev is None:
            xd,zd = 0.,0.
        else:
            dt=t-self.prev[0]
            if dt <= 0:
                raise RuntimeError('仿真时间发生重置')
            xd,zd = (xi-self.prev[1])/dt,(zeta-self.prev[2])/dt
        self.prev=(t,xi,zeta)
        a,w=d['imu'].linear_acceleration,d['imu'].angular_velocity
        p,q=d['pose'].pose.position,d['pose'].pose.orientation
        v=d['vel'].twist.linear
        writer.writerow([t,a.x,a.y,a.z,w.x,w.y,w.z,p.x,p.y,p.z,v.x,v.y,v.z,q.w,q.x,q.y,q.z,
                         d['cmd'].thrust,xi,zeta,xd,zd,lp.x,lp.y,lp.z,dp.x,dp.y,dp.z,
                         d['imu'].header.stamp.to_sec(),d['pose'].header.stamp.to_sec(),
                         d['cmd'].header.stamp.to_sec(),age,gq.w,gq.x,gq.y,gq.z,gv.x,gv.y,gv.z])
        self.rows += 1

    def run(self, output, duration):
        deadline=time.monotonic()+60
        while not rospy.is_shutdown():
            if 'state' in self.msg and self.msg['state'].connected and 'pose' in self.msg and 'links' in self.msg:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('等待 SITL 遥测超时')
            time.sleep(.1)
        assert 'iris::payload' in self.msg['links'].name
        p=self.msg['pose'].pose.position
        x0,y0,z0=p.x,p.y,p.z
        if abs(z0)>10 or abs(x0)>10 or abs(y0)>10:
            raise RuntimeError('飞控位置初始化异常：{}'.format((x0,y0,z0)))
        target=z0+3.0
        rate=rospy.Rate(50)
        print('起飞准备：local origin={}, target_z={}'.format((x0,y0,z0),target),flush=True)
        for _ in range(150):
            self.setpoint(x0,y0,target)
            rate.sleep()
        deadline=time.monotonic()+90
        request=0
        stable=0
        stand_removed=False
        while not rospy.is_shutdown():
            self.setpoint(x0,y0,target)
            now=time.monotonic()
            s=self.msg['state']
            if now-request>3:
                request=now
                if s.mode!='OFFBOARD':
                    print('OFFBOARD:',self.mode(custom_mode='OFFBOARD').mode_sent,flush=True)
                elif not s.armed:
                    print('ARM:',self.arm(True).success,flush=True)
            p=self.msg['pose'].pose.position
            if s.armed and not stand_removed and p.z>z0+.3:
                reply=rospy.ServiceProxy('/gazebo/delete_model',DeleteModel)('takeoff_stand')
                if not reply.success:
                    raise RuntimeError('移除临时起飞支架失败：'+reply.status_message)
                stand_removed=True
                print('机体离架，已移除临时支架以避免负载碰撞',flush=True)
            if s.armed and s.mode=='OFFBOARD' and abs(p.z-target)<.25:
                stable+=1
            else:
                stable=0
            if stable>=150:
                break
            if now>deadline:
                raise TimeoutError('起飞未完成；不绕过飞控检查')
            rate.sleep()
        print('稳定悬停，开始采集 {} 秒新轨迹'.format(duration),flush=True)
        start=rospy.get_time()
        deadline=time.monotonic()+duration*3+60
        next_report=0
        with output.open('x',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle)
            writer.writerow(HEADER)
            while not rospy.is_shutdown():
                t=rospy.get_time()-start
                if t>=duration:
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('采集墙钟超时')
                tx,ty = trajectory(t, self.config)
                x,y=x0+tx,y0+ty
                self.setpoint(x,y,target)
                state=self.msg['state']
                if not state.armed or state.mode!='OFFBOARD':
                    raise RuntimeError('飞行控制状态异常，中止正式采集')
                p=self.msg['pose'].pose.position
                if abs(p.x-x0)>15 or abs(p.y-y0)>15 or abs(p.z-target)>4:
                    raise RuntimeError('仿真轨迹偏离限制')
                links=self.msg['links']
                dp=links.pose[links.name.index('iris::base_link')].position
                lp=links.pose[links.name.index('iris::payload')].position
                angle=math.degrees(math.atan2(math.hypot(lp.x-dp.x,lp.y-dp.y),dp.z-lp.z))
                if angle>70 or lp.z<.4:
                    raise RuntimeError('吊载倾角超过70度或对地间隙不足，中止正式采集')
                self.sample(writer)
                if t>=next_report:
                    print('采集 {:.1f}/{} s，{} 行，跳过 {} 行'.format(t,duration,self.rows,self.skips),flush=True)
                    handle.flush()
                    next_report+=20
                rate.sleep()
        if rospy.is_shutdown() or self.rows < duration*50*.95:
            raise RuntimeError('采集未完整结束或有效行数不足95%')
        # Return to origin, then land on the ground (support was removed).
        for _ in range(250):
            self.setpoint(x0,y0,target)
            rate.sleep()
        print('AUTO.LAND:',self.mode(custom_mode='AUTO.LAND').mode_sent,flush=True)
        deadline=time.monotonic()+60
        while self.msg['state'].armed and time.monotonic()<deadline:
            time.sleep(.2)
        return {'rows':self.rows,'skipped':self.skips,'start_sim_time':start,
                'landed_disarmed':not self.msg['state'].armed,'origin_local':[x0,y0,z0], 'target_z':target}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=180)
    parser.add_argument('--sdf',type=Path,default=Path('/mnt/d/RPi_Deployment/EKF-LSTM_for_Swing_Angle_Estimation-main/simulation/models/iris_slung_load/iris_slung_load.sdf'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if os.environ.get('ROS_MASTER_URI','').rstrip('/')!='http://127.0.0.1:11321':
        raise RuntimeError('仅允许本任务隔离的本地仿真 ROS master')
    if args.duration != 180:
        raise ValueError('本协议固定180秒，避免改变预先指定的阶段')
    output=args.output.resolve()
    metadata=output.with_suffix('.json')
    if output.exists() or metadata.exists():
        raise FileExistsError('目标飞行文件已存在')
    output.parent.mkdir(parents=True,exist_ok=True)
    rospy.init_node('independent_sitl_flight')
    if not rospy.get_param('/use_sim_time',False):
        raise RuntimeError('必须使用仿真时间')
    rospy.wait_for_service('/mavros/set_message_interval',timeout=30)
    reply=rospy.ServiceProxy('/mavros/set_message_interval',MessageInterval)(message_id=83,message_rate=50)
    if not reply.success:
        raise RuntimeError('无法设置ATTITUDE_TARGET推力遥测为50Hz')
    flight=Flight(config)
    report={'status':'running','duration_s':args.duration,'rate_hz':50,
            'trajectory':'12s hover; four 36s mixed stages; 24s hover; cosine ramps',
            'flight_config':config,
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'trajectory_code_sha256':hashlib.sha256(Path(__file__).with_name('轨迹协议.py').read_bytes()).hexdigest(),
            'requested_attitude_target_telemetry_hz':50,
            'truth':'repository projection angles from Gazebo world relative link positions',
            'model':str(args.sdf.resolve()),
            'model_sha256':hashlib.sha256(args.sdf.read_bytes()).hexdigest(),
            'world_sha256':hashlib.sha256(Path('/mnt/d/RPi_Deployment/离线测试/独立仿真实验/吊载起飞场景.world').read_bytes()).hexdigest(),
            'takeoff_support':'deleted after vehicle rises 0.3 m; absent during recording',
            'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        report.update(flight.run(output,args.duration))
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),rows=flight.rows)
        if 'state' in flight.msg and flight.msg['state'].armed:
            try:
                flight.mode(custom_mode='AUTO.LAND')
            except Exception:
                pass
        raise
    finally:
        if output.exists():
            report['csv_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
        metadata.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report,flush=True)


if __name__=='__main__':
    main()
