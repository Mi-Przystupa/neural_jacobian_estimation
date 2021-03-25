import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class SerialTwoDOFGym(gym.core.Env):
    def __init__(   self,
                    L1=0.3143,
                    L2=0.1674 + 0.120,
                    H = 100,
                    dt = 0.05,
                    max_action = np.array([0.5, 0.5]),
                    angle_boundary_low = np.array([- np.pi/4, - 3*np.pi/4]),
                    angle_boundary_high = np.array([+ np.pi/4, + 3*np.pi/4]),
                    position_boundary_low = np.array([0,-0.5]),
                    position_boundary_high = np.array([0.7,0.5]),
                    target_angle_boundary_low = np.array([- np.pi/4, -3*np.pi/4]),
                    target_angle_boundary_high = np.array([ np.pi/4, 3*np.pi/4]),
                    target_xy_boundary_low = np.array([0,-0.5]),
                    target_xy_boundary_high = np.array([0.6,0.5]),
                    target_mode = 'position',
                    boundary_mode = 'position',
                    obs_mode = 'complete'):
        #robot parameter stuff
        self.L1 = L1
        self.L2 = L2
        self.th1 = 0.0
        self.th2 = 0.0
        self.dt = dt#0.04 #change
        self.max_action = max_action#np.array([0.1, 0.1])
        self.ang_v = np.array([0.0, 0.0])
        self.base_psn = np.array([0, 0, 0, 1])
        #just let them be set in update_psn
        self.l1_psn, self.l2_psn  = 0.0, 0.0
        self.update_psn()

        self.boundary_mode = boundary_mode
        self.angle_boundary_low = angle_boundary_low
        self.angle_boundary_high = angle_boundary_high
        self.position_boundary_low = position_boundary_low
        self.position_boundary_high = position_boundary_high

        #set-up target
        self.target_mode = target_mode
        self.target_angle_boundary_low = target_angle_boundary_low
        self.target_angle_boundary_high = target_angle_boundary_high
        self.target = np.copy(self.base_psn) #otherwise point to same thing
        self.target_xy_boundary_low = target_xy_boundary_low
        self.target_xy_boundary_high = target_xy_boundary_high
        self.setTarget()

        #rendering configuration things
        self.fig = None
        self.ax = None
        self.robot = None
        self.targImg = None

        #iterations for episode
        self.H = H
        self.steps = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # [current_pos, target, current_speed]
        self.obs_mode = obs_mode
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)

    def _T(self, th, L):
        c = np.cos(th)
        s = np.sin(th)
        return np.array([
            [ c,-s, 0, L],
            [ s, c, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
        ])

    def setTarget(self):
        if self.target_mode == 'position':
            while True:
                A = np.random.uniform(  low=self.target_xy_boundary_low,
                                        high=self.target_xy_boundary_high)
                L = np.linalg.norm(A)
                if L <= (self.L1 + self.L2) and L >= np.abs(self.L1 - self.L2):
                    (th1s, th2s) = self.inv_kin(A[0], A[1])
                    if self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, np.array([th1s[0], th2s[0]]), 0, self.dt) or self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, np.array([th1s[1], th2s[1]]), 0, self.dt):
                        self.target = A
                        break

        elif self.target_mode == 'angular':

            org_th = self.th1
            org_th2 = self.th2

            A = np.random.uniform(low=self.target_angle_boundary_low, high=self.target_angle_boundary_high)

            self.th1 = A[0]
            self.th2 = A[1]

            T02 = np.matmul(self.T01(),self.T12())
            T0e = np.matmul(T02, self.T2e())
            self.target = np.matmul(T0e, self.target)
            self.th1 = org_th
            self.th2 = org_th2

    def T01(self):
        return self._T(self.th1, 0)
    def T12(self):
        return self._T(self.th2, self.L1)
    def T2e(self):
        return self._T(0.0, self.L2)

    def update_psn(self):
        T02 = np.matmul(self.T01(),self.T12())
        T0e = np.matmul(T02, self.T2e())
        self.l1_psn = np.matmul(T02, self.base_psn)
        self.l2_psn = np.matmul(T0e, self.base_psn)

    def _obs(self):

        x, y = self.l2_psn[0], self.l2_psn[1]
        th1, th2 = self.th1, self.th2
        th1v, th2v = self.ang_v[0], self.ang_v[1]
        if self.obs_mode == 'complete':
            return np.array([x, y, np.cos(th1), np.sin(th1),  np.cos(th2), np.sin(th2), th1v, th2v, self.target[0], self.target[1]])
        elif self.obs_mode == 'reduced':
            return np.array([th1, th2, th1v, th2v,self.target[0]-x, self.target[1]-y]) #change
    def _obs2(self):
        #return p
        x, y = self.l2_psn[0], self.l2_psn[1]
        th1, th2 = self.th1, self.th2
        th1v, th2v = self.ang_v[0], self.ang_v[1]
        return np.array([th1, th2, th1v, th2v,self.target[0]-x, self.target[1]-y]) #change

    # def com
    def _calcReward1(self):
        diff = self.target[0:2] - self.l2_psn[0:2]
        dist = np.linalg.norm(diff, 2)
        return self.dt * (-dist + np.exp(-(dist**2)/0.01))

    def _calcReward2(self):
        th1,th2 = self.inv_kin(self.target[0], self.target[1])
        d1 = np.array([th1[0]-self.th1, th2[0]-self.th2])
        d2 = np.array([th1[1]-self.th1, th2[1]-self.th2])
        d1[np.where(d1>np.pi)] -= 2*np.pi
        d1[np.where(d1<-np.pi)] += 2*np.pi
        d2[np.where(d2>np.pi)] -= 2*np.pi
        d2[np.where(d2<-np.pi)] += 2*np.pi
        d1_time = np.max(np.abs(d1)/self.max_action)
        d2_time = np.max(np.abs(d2)/self.max_action)
        T = min([d1_time, d2_time])
        diff = d1 * (d1_time < d2_time) + d2*(d1_time>= d2_time)

        return -T*self.dt

    def _calcReward3(self):
        th1,th2 = self.inv_kin(self.target[0], self.target[1])
        d1 = np.array([th1[0]-self.th1, th2[0]-self.th2])
        d2 = np.array([th1[1]-self.th1, th2[1]-self.th2])
        d1[np.where(d1>np.pi)] -= 2*np.pi
        d1[np.where(d1<-np.pi)] += 2*np.pi
        d2[np.where(d2>np.pi)] -= 2*np.pi
        d2[np.where(d2<-np.pi)] += 2*np.pi
        cost = min(np.linalg.norm(d1), np.linalg.norm(d2) )
        return -cost*self.dt

    def _calcReward3(self):
        th1,th2 = self.inv_kin(self.target[0], self.target[1])
        d1 = np.array([th1[0]-self.th1, th2[0]-self.th2])
        d2 = np.array([th1[1]-self.th1, th2[1]-self.th2])
        d1[np.where(d1>np.pi)] -= 2*np.pi
        d1[np.where(d1<-np.pi)] += 2*np.pi
        d2[np.where(d2>np.pi)] -= 2*np.pi
        d2[np.where(d2<-np.pi)] += 2*np.pi
        cost = min(np.linalg.norm(d1), np.linalg.norm(d2) )
        return -cost*self.dt

    def step(self, a=None):
        # clipping the action
        a = np.clip(a,-1,1) * self.max_action
        # a = np.minimum( np.maximum(-self.max_action, a), self.max_action)

        if a is not None:
            self.ang_v = a
        if self.boundary_mode == 'angular':
            in_boundary_0 = self.check_boundary(low=self.angle_boundary_low[0], high=self.angle_boundary_high[0], current=self.th1, rate=a[0], dt=self.dt)
            in_boundary_1 = self.check_boundary(low=self.angle_boundary_low[1], high=self.angle_boundary_high[1], current=self.th1, rate=a[1], dt=self.dt)
            self.ang_v = [in_boundary_0 * a[0], in_boundary_1*a[1] ]
        elif self.boundary_mode == 'position':
            n = self.forward(th1=self.th1+a[0]*self.dt, L1=self.L1, th2=self.th2+a[1]*self.dt, L2=self.L2)
            in_boundary = self.check_boundary(self.position_boundary_low, self.position_boundary_high, n[:2],0,0)
            self.ang_v = in_boundary * a
        # if self.in_boundary:
        #     self.ang_v = a
        # else :
        #     self.ang_v = [0,0]

        self.th1 += self.ang_v[0] * self.dt
        self.th2 += self.ang_v[1] * self.dt

        self.update_psn()

        observation = self._obs()
        reward = self._calcReward1()
        self.steps += 1
        if self.steps == self.H:
            done = True
        else:
            done = False

        # return s, r, t, {}
        return observation, reward, done, None

    def reset(self):
        #reset robot position
        (th1,th2)=self.inv_kin(0.4,0)
        self.th1, self.th2 = th1[0], th2[0]
        self.ang_v = np.array([0., 0.])
        self.update_psn()
        #change target
        self.target = np.copy(self.base_psn) #otherwise point to same thing
        self.setTarget()

        # reset steps
        self.steps = 0

        return self._obs()

    def _initFig(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        x = self.L1 + self.L2
        y = self.L1 + self.L2
        self.ax.set(xlim=(-x, x), ylim=(-y, y))
        self.ax.set_xticks([])
        self.ax.set_yticks([])


    def render(self, mode='human'):
        if self.fig is None and self.ax is None:
            self._initFig()
        def joint_dim(i):
            return [self.base_psn[i], self.l1_psn[i], self.l2_psn[i]]
        x = joint_dim(0)
        y = joint_dim(1)

        if self.targImg is None:
            self.targImg = self.ax.scatter([self.target[0]], [self.target[1]])
        else:
            self.targImg.set_offsets(np.c_[self.target[0], self.target[1]])


        if self.robot is None:
            self.robot,  = plt.plot(x, y)
        else:
            self.robot.set_xdata(x)
            self.robot.set_ydata(y)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return None

    def forward(self,th1,L1, th2,L2):
        return self._T(th1, 0).dot(self._T(th2, L1).dot(self._T(0, L2))).dot(self.base_psn)

    @staticmethod
    def check_boundary(low, high, current, rate, dt):
        safety_coefficient = 1
        next = current + safety_coefficient * rate * dt
        if np.all( next >= low) and np.all( next <= high):
            return True

        return False
    def inv_kin(self,tx,ty):
        l1 = self.L1
        l2 = self.L2
        l3 = (tx**2+ty**2)**0.5
        th0 = np.arctan2(ty,tx)
        a = np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
        th1 = np.array([a, -a])+th0
        a = np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2))
        th2 = np.array([a-np.pi, np.pi-a])
        return (th1,th2)

