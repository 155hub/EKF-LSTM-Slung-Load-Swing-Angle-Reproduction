import numpy as np
import math

class SlungLoadEKF:
    def __init__(self, m_drone=1.5, m_load=0.5, l_rope=2.0):
        """
        初始化 EKF
        :param m_drone: 无人机质量 (kg), Iris 约为 1.5kg [cite: 233]
        :param m_load: 负载质量 (kg), 你的设定为 0.5kg
        :param l_rope: 绳长 (m), 你的设定为 2.0m
        """
        # --- 1. 物理参数 ---
        self.m = m_drone       # m
        self.m_l = m_load      # m_l
        self.L = l_rope        # L
        self.M = self.m + self.m_l  # 总质量 M [cite: 146]
        self.g = 9.81

        # --- 2. 状态向量 x (7x1) ---
        # x = [xi, zeta, xi_dot, zeta_dot, fax, fay, faz]
        # 对应论文中的 x1 到 x7 [cite: 162]
        self.x = np.zeros(7)

        # --- 3. 协方差矩阵 P (7x7) ---
        # 初始化为较小的值，参考论文 Case 1 或 Case 2 的设定
        self.P = np.eye(7) * 1e-4

        # --- 4. 噪声矩阵 Q (过程噪声) 和 R (测量噪声) ---
        # 参数参考 Section 4.1.1 [cite: 273]
        # 状态: [角度, 角度, 角速度, 角速度, 力x, 力y, 力z]
        # 论文设定: Q = diag(10^-7, ..., 1, 1, 10^-7)
        self.Q = np.diag([
            1e-6, 1e-6,   # 角度 (xi, zeta)
            1e-4, 1e-4,   # 角速度 (xi_dot, zeta_dot) - 允许一定变化，但不要太剧烈
            1e-4, 1e-4, 1e-4 # 干扰力 - 适当减小，避免把所有误差都算作干扰
        ])
        
        # 测量: [ax, ay, az]
        # 论文设定: R = 3.6e-5 * I_3
        self.R = np.eye(3) * 0.002

    def _compute_state_derivative(self, x, u):
        """
        计算状态导数 dx/dt = f(x, u)
        :param x: 状态向量 [xi, zeta, xi_dot, zeta_dot, fax, fay, faz]
        :param u: 控制输入 [u1, u2, u3]
        :return: 状态导数 dx (7x1)
        """
        # 提取状态变量
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        x5, x6, x7 = x[4], x[5], x[6]
        u1, u2, u3 = u[0], u[1], u[2]
        
        c1, s1 = np.cos(x1), np.sin(x1)
        c2, s2 = np.cos(x2), np.sin(x2)
        
        # 非线性状态方程 f(x, u)
        # 公式来源: Eq (8) [cite: 164-170]
        dx = np.zeros(7)
        
        # 摆角导数 = 角速度
        dx[0] = x3 
        dx[1] = x4
        
        # 角加速度 (核心动力学)
        # 论文 Eq (8) 第3行
        term_xi_num = (u2*c1 + x6*c1 + u3*s1 + x7*s1 + 2*self.L*self.m*x3*x4*s2)
        term_xi_den = (self.L * self.m * c2)
        if abs(term_xi_den) < 1e-6: 
            term_xi_den = 1e-6  # 防止除零
        dx[2] = term_xi_num / term_xi_den
        
        # 论文 Eq (8) 第4行 (注意负号)
        term_zeta_num = (self.L * self.m * c2 * s2 * (x3**2) + 
                         u1 * c2 + x5 * c2 - 
                         u3 * c1 * s2 - x7 * c1 * s2 + 
                         u2 * s1 * s2 + x6 * s1 * s2)
        dx[3] = - (term_zeta_num) / (self.L * self.m)
        
        # 干扰力假设为常数 (Random Walk)
        dx[4] = 0
        dx[5] = 0
        dx[6] = 0
        
        return dx

    def predict(self, u, dt):
        """
        EKF 预测步骤 (Time Update)
        :param u: 控制输入力向量 [u1, u2, u3] (在 NED 惯性系下) 
        :param dt: 时间步长 (s)
        """
        # --- A. 使用 RK4 积分状态方程 ---
        # 论文推荐使用 RK4 [cite: 227]
        k1 = self._compute_state_derivative(self.x, u)
        k2 = self._compute_state_derivative(self.x + 0.5*dt*k1, u)
        k3 = self._compute_state_derivative(self.x + 0.5*dt*k2, u)
        k4 = self._compute_state_derivative(self.x + dt*k3, u)
        
        self.x = self.x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- B. 计算状态转移 Jacobian F ---
        # 公式来源: Appendix A (A1 - A13) [cite: 969-1002]
        # 需要使用更新后的状态来计算 Jacobian
        x1, x2, x3, x4 = self.x[0], self.x[1], self.x[2], self.x[3]
        x5, x6, x7 = self.x[4], self.x[5], self.x[6]
        u1, u2, u3 = u[0], u[1], u[2]
        
        c1, s1 = np.cos(x1), np.sin(x1)
        c2, s2 = np.cos(x2), np.sin(x2)
        tan2 = np.tan(x2)
        
        # 辅助计算
        Lm = self.L * self.m
        Lmc2 = Lm * c2
        Lmc2_sq = Lm * (c2**2)
        
        # 填充 F 矩阵 (连续时间 Jacobian)
        Fc = np.zeros((7, 7))
        Fc[0, 2] = 1.0 # F13
        Fc[1, 3] = 1.0 # F24
        Fc[2, 0] = (u3*c1 + x7*c1 - u2*s1 - x6*s1) / Lmc2
        Fc[2, 1] = 2*x3*x4 + (s2 * (u2*c1 + x6*c1 + u3*s1 + x7*s1 + 2*Lm*x3*x4*s2)) / Lmc2_sq
        Fc[2, 2] = 2 * x4 * tan2
        Fc[2, 3] = 2 * x3 * tan2
        Fc[2, 5] = c1 / Lmc2
        Fc[2, 6] = s1 / Lmc2
        Fc[3, 0] = -s2 * ((u2 + x6)*c1 + (u3 + x7)*s1) / Lm
        Fc[3, 1] = (u1*s2 + x5*s2 + Lm*(x3**2) + u3*c1*c2 + x7*c1*c2 - 
                    u2*c2*s1 - x6*c2*s1 - 2*self.m*(x3**2)*(c2**2)) / Lm
        Fc[3, 2] = -x3 * np.sin(2*x2)
        Fc[3, 4] = -c2 / Lm
        Fc[3, 5] = -s1 * s2 / Lm
        Fc[3, 6] = c1 * s2 / Lm
        
        # 离散化 P 更新: P = F_k * P * F_k^T + Q
        # F_k = I + Fc * dt
        F_k = np.eye(7) + Fc * dt
        self.P = F_k @ self.P @ F_k.T + self.Q

    def update(self, z_meas):
        """
        EKF 更新步骤 (Measurement Update)
        :param z_meas: 测量的加速度 [ax, ay, az] (在 NED 惯性系下) [cite: 192]
        """
        if not hasattr(self, 'u_last'):
             # 如果没有 u，无法计算 h(x)，直接返回
             return 
        
        # 提前定义相关参数
        # h(x, u)使用：
        x1, x2, x3, x4 = self.x[0], self.x[1], self.x[2], self.x[3]
        x5, x6, x7 = self.x[4], self.x[5], self.x[6]
        u1, u2, u3 = self.u_last[0], self.u_last[1], self.u_last[2]
        # H额外使用：
        c1, s1 = np.cos(x1), np.sin(x1)
        c2, s2 = np.cos(x2), np.sin(x2)
        mM = self.m * self.M

        m, ml, M, L = self.m, self.m_l, self.M, self.L
        c1_sq, s1_sq = c1**2, s1**2
        c2_sq, s2_sq = c2**2, s2**2
        c2_3 = c2**3          # cos^3(x2)
        s2_2 = 2 * s2 * c2    # sin(2*x2)
        
        # --- A. 观测模型 h(x, u) ---
        # 公式来源: Eq (9) [cite: 172-185]
        # 计算理论上的加速度 (Predicted Measurement)
        h_x = self._h_model(self.x, self.u_last)

        # --- B. 计算观测 Jacobian H ---
        # 公式来源: Appendix A (A14 - A34) [cite: 1003-1078]
        # 这是一个 3x7 矩阵
        H = np.zeros((3, 7))
        
        # H11 (A14)
        # [cite: 1003]
        term_h11 = (u2 + x6) * c1 + (u3 + x7) * s1
        H[0, 0] = (ml * s2_2 * term_h11) / (2 * mM)
        t12_1 = (u1 + x5) * s2_2
        t12_2 = -u3 * c1 - x7 * c1 + u2 * s1 + x6 * s1
        t12_3 = 2 * (u3 + x7) * c1 * c2_sq
        t12_4 = -2 * (u2 + x6) * c2_sq * s1
        t12_5 = -3 * m * (x3**2) * c2_3 + 2 * m * (x3**2) * c2
        t12_6 = -L * m * (x4**2) * c2
        H[0, 1] = -ml * (t12_1 + t12_2 + t12_3 + t12_4 + t12_5 + t12_6) / mM
        H[0, 2] = (2 * L * ml * x3 * s2 * c2_sq) / M
        H[0, 3] = (2 * L * ml * x4 * s2) / M
        H[0, 4] = (m + ml * c2_sq) / mM
        H[0, 5] = (ml * c2 * s1 * s2) / mM
        H[0, 6] = -(ml * c1 * c2 * s2) / mM

        t21_1 = (u3 + x7) * c2 - (u1 + x5) * c1 * s2
        t21_2 = -2 * (u3 + x7) * c1_sq * c2
        t21_3 = 2 * (u2 + x6) * c1 * c2 * s1
        t21_4 = L * m * (x4**2) * c1 + L * m * (x3**2) * c1 * c2_sq
        H[1, 0] = -ml * c2 * (t21_1 + t21_2 + t21_3 + t21_4) / mM
        t22_1 = (u1 + x5) * s1 - (u2 + x6) * s2_2
        t22_2 = -2 * (u1 + x5) * c2_sq * s1
        t22_3 = 2 * (u2 + x6) * c1_sq * c2 * s2
        t22_4 = 2 * (u3 + x7) * c1 * c2 * s1 * s2
        t22_5 = -L * m * (x4**2) * s1 * s2
        t22_6 = 3 * m * (x3**2) * s1 * s2 * (s2_sq - 1) # (sin^2 - 1) = -cos^2
        H[1, 1] = -ml * (t22_1 + t22_2 + t22_3 + t22_4 + t22_5 + t22_6) / mM
        H[1, 2] = -(2 * L * ml * x3 * c2_3 * s1) / M
        H[1, 3] = -(2 * L * ml * x4 * c2 * s1) / M
        H[1, 4] = (ml * c2 * s1 * s2) / mM
        H[1, 5] = (m + ml - ml * c2_sq + ml * c1_sq * c2_sq) / mM
        H[1, 6] = (ml * c1 * c2_sq * s1) / mM

        t31_1 = (u1 + x5) * s1 * s2 - (u2 + x6) * c2
        t31_2 = 2 * (u2 + x6) * c1_sq * c2
        t31_3 = 2 * (u3 + x7) * c1 * c2 * s1
        t31_4 = -L * m * (x4**2) * s1 - L * m * (x3**2) * c2_sq * s1
        H[2, 0] = (ml * c2 * (t31_1 + t31_2 + t31_3 + t31_4)) / mM
        t32_inner = 2 * (u1 + x5) * c2_sq - (u1 + x5)
        t32_bracket = (t32_inner + L * m * (x4**2) * s2 
                       - 2 * (u3 + x7) * c1 * c2 * s2 
                       + 2 * (u2 + x6) * c2 * s1 * s2 
                       + 3 * m * (x3**2) * c2_sq * s2)
        H[2, 1] = -(ml * c1 * t32_bracket) / mM
        H[2, 2] = (2 * L * ml * x3 * c1 * c2_3) / M
        H[2, 3] = (2 * L * ml * x4 * c1 * c2) / M
        H[2, 4] = -(ml * c1 * c2 * s2) / mM
        H[2, 5] = (ml * c1 * c2_sq * s1) / mM
        H[2, 6] = (M - ml * c1_sq * c2_sq) / mM

        # # 实际工程中，也可以使用数值微分法 (Numerical Differentiation) 来计算 H，以避免手写公式出错。
        # epsilon = 1e-6
        # H_num = np.zeros((3, 7))
        # x_curr = self.x.copy()
        # h_curr = self._h_model(x_curr, self.u_last) # 封装上面的 h_x 计算逻辑
        
        # for i in range(7):
        #     x_pert = x_curr.copy()
        #     x_pert[i] += epsilon
        #     h_pert = self._h_model(x_pert, self.u_last)
        #     H_num[:, i] = (h_pert - h_curr) / epsilon
        
        # H = H_num # 使用数值 Jacobian 替代解析解

        # --- C. 卡尔曼更新 ---
        # 1. 计算 S = H P H^T + R
        PHT = self.P @ H.T
        S = H @ PHT + self.R
        
        # [保护] 检查 S 是否含有 NaN 或 Inf
        if np.any(np.isnan(S)) or np.any(np.isinf(S)):
            print("[EKF Error] S matrix exploded! Resetting Covariance.")
            self.P = np.eye(7) * 1.0  # 重置 P 矩阵
            return

        # 2. 计算 K = P H^T S^-1
        try:
            # 使用伪逆 (pinv) 比 inv 更稳定，防止 S 接近奇异矩阵
            K = PHT @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            print("[EKF Error] S matrix inversion failed.")
            return

        # 3. 计算 Innovation
        y = z_meas - h_x
        
        # [保护] 限制 Innovation 的大小 (防止异常测量值导致发散)
        # 例如限制残差在 +/- 10 m/s^2 以内，超过则认为是野值
        y = np.clip(y, -10.0, 10.0)

        # 4. 更新状态 x = x + K y
        self.x = self.x + K @ y
        
        # 5. 更新协方差 P = (I - K H) P
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P
        
        # [关键] 强制 P 矩阵对称 (数值稳定性)
        self.P = 0.5 * (self.P + self.P.T)
        
        # [关键] 强制对角线为正 (防止协方差变为负数)
        # 这一步能有效防止 NaN 的产生
        for i in range(7):
            if self.P[i, i] < 0:
                self.P[i, i] = 1e-6

    def set_current_control(self, u):
        """记录当前的控制力，供 update 步骤使用"""
        self.u_last = u

    def _h_model(self, x, u):
        """辅助函数：计算 h(x, u)，用于数值 Jacobian"""
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        x5, x6, x7 = x[4], x[5], x[6]
        u1, u2, u3 = u[0], u[1], u[2]
        
        c1, s1 = np.cos(x1), np.sin(x1)
        c2, s2 = np.cos(x2), np.sin(x2)
        mM = self.m * self.M

        h_x = np.zeros(3)
        
        # h1 (x-axis accel)
        term1 = u1*self.m + x5*self.m + u1*self.m_l*(c2**2) + x5*self.m_l*(c2**2)
        term2 = self.L*self.m*self.m_l*(x4**2)*s2
        term3 = -u3*self.m_l*c1*c2*s2 - x7*self.m_l*c1*c2*s2
        term4 = u2*self.m_l*c2*s1*s2 + x6*self.m_l*c2*s1*s2
        term5 = self.L*self.m*self.m_l*(x3**2)*(c2**2)*s2
        h_x[0] = (term1 + term2 + term3 + term4 + term5) / mM
        
        # h2 (y-axis accel)
        term1 = u2*self.m + x6*self.m + u2*self.m_l + x6*self.m_l 
        term2 = -u2*self.m_l*(c2**2) - x6*self.m_l*(c2**2)
        term3 = u2*self.m_l*(c1**2)*(c2**2) + x6*self.m_l*(c1**2)*(c2**2)
        term4 = u3*self.m_l*c1*(c2**2)*s1 + x7*self.m_l*c1*(c2**2)*s1
        term5 = u1*self.m_l*c2*s1*s2 + x5*self.m_l*c2*s1*s2
        term6 = -self.L*self.m*self.m_l*(x4**2)*c2*s1 - self.L*self.m*self.m_l*(x3**2)*(c2**3)*s1
        h_x[1] = (term1 + term2 + term3 + term4 + term5 + term6) / mM

        # h3 (z-axis accel)
        # 注意：论文 Eq(9) 最后一个分量包含 +g terms，但 IMU 测量通常包含重力或者需要减去重力
        # 论文 Eq(1) 和 Eq(2) 定义 m*v_dot = fg + ... 
        # 这里的 h(x) 算的是 inertial acceleration v_dot。
        # 如果 z_meas 是来自 IMU (specific force)，则 measured_acc = v_dot - g。
        # 需要根据 PX4 输出确认。通常 local_position/accel 是运动加速度 (无重力)。
        # 假设 z_meas 是纯运动加速度 (kinematic acceleration)。
        term1 = u3*self.m + x7*self.m + u3*self.m_l + x7*self.m_l + self.g*(self.m**2) + self.g*self.m*self.m_l
        term2 = -u3*self.m_l*(c1**2)*(c2**2) - x7*self.m_l*(c1**2)*(c2**2)
        term3 = u2*self.m_l*c1*(c2**2)*s1 + x6*self.m_l*c1*(c2**2)*s1
        term4 = -u1*self.m_l*c1*c2*s2 - x5*self.m_l*c1*c2*s2
        term5 = self.L*self.m*self.m_l*(x4**2)*c1*c2 + self.L*self.m*self.m_l*(x3**2)*c1*(c2**3)
        h_x[2] = (term1 + term2 + term3 + term4 + term5) / mM
        return h_x

    def get_swing_state(self):
        """返回当前的摆角 (deg) 和角速度 (deg/s)"""
        return np.degrees(self.x[0:2]), np.degrees(self.x[2:4])