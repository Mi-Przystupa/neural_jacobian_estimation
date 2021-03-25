from environments.kinova_env import KinovaEnv, Bound
from gym import spaces, Wrapper
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import copy

class HalfSphereSampler:
    def __init__(self, radius_bound, theta_bound, phi_bound):
        self.radius_bound = Bound(low=radius_bound[0], high=radius_bound[1])
        self.phi_bound = Bound(low=phi_bound[0], high=phi_bound[1])
        self.theta_bound = Bound(low=theta_bound[0], high=theta_bound[1])

    def sample(self):
        r = np.random.uniform(self.radius_bound.low, self.radius_bound.high)
        theta = np.random.uniform(self.theta_bound.low, self.theta_bound.high)
        #note that the lower bound is for angles closer to origin, and higher is for further away)
        u = np.random.uniform(self.phi_bound.low, self.phi_bound.high)

        phi = np.arccos(1-u)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return [x, y, z]

class CubeSampler:
    def __init__(self, x_bound, y_bound, z_bound):
        self.x_bound = Bound(low=x_bound[0], high=x_bound[1])
        self.y_bound = Bound(low=y_bound[0], high=y_bound[1])
        self.z_bound = Bound(low=z_bound[0], high=z_bound[1])
    def sample(self):
        x = np.random.uniform(self.x_bound.low, self.x_bound.high)
        y = np.random.uniform(self.y_bound.low, self.y_bound.high)
        z = np.random.uniform(self.z_bound.low, self.z_bound.high)
        return [x, y, z]


class SimulatorKinovaGripper(KinovaEnv):
    VALID_REWARDS = ['l2', 'l1', 'squared', 'precision', 'action-norm', 'discrete-time-penalty', 'keep-moving']
    VALID_TARGET_GENERATION = ['half-sphere', 'kinematic', 'cube', 'fixed']

    def __init__(self,
                    dt=0.05,
                    max_velocity=None,
                    control_joints = list(range(7)),
                    joint_bounds=None,
                    H=100,
                    reward_type='l2',
                    precision_tau=0.01,
                    epsilon_threshold=0.1,
                    target_generation = 'kinematic',
                    radius_bound = [0.6, 1.07], #lower bound is psuedo arbitrary
                    phi_bound =[0.0, 1.0], #to be be more cone like shrink upper bound
                    theta_bound = [0.0, np.pi * 2],
                    fixed_target = np.array([0.6, 0.6, 0.6])
                    ):

        super(SimulatorKinovaGripper, self).__init__(ths=np.zeros(7),
                                                dt=dt, use_gripper=True, max_velocity=max_velocity,
                                                H = H)


        self._link_psns = self.kinematics.calc_link_positions()
        self._control_joints = control_joints

        self.ang_v = np.zeros(len(control_joints))

        self.joint_bounds = Bound([l.low for l in self.JOINT_BOUNDS],
                [l.high for l in self.JOINT_BOUNDS]) if joint_bounds is None else joint_bounds

        #hard coded for now
        self.boundary_mode = 'angular'
        self.obs_mode = 'complete'


        self.target_generation = target_generation.lower()
        self.fixed_target = fixed_target
        self.target = np.zeros(3) if  self.target_generation != 'fixed' else fixed_target
        #TODO: the things below allow samples for invalid positions of robot
        #just like FYI that this does exist....
        self.target_sampler = None
        if self.target_generation == 'half-sphere':
            self.target_sampler = HalfSphereSampler(radius_bound, theta_bound, phi_bound)
        elif self.target_generation == 'cube':
            #TODO fix this.... to not be internally handled
            length = self.TOTAL_LENGH / self.MILLIMETER_TO_METERS
            bound = [-length, length]
            self.target_sampler = CubeSampler(bound, bound, bound)

        #OpenAI gym configs
        self.action_space = spaces.Box(low=-self._max_velocity
                , high=self._max_velocity, shape=(len(control_joints),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)

        self.reward_type = reward_type.lower().split(',')

        for r_t in self.reward_type:
            assert r_t in self.VALID_REWARDS

        self.precision_tau = precision_tau
        self.epsilon_threshold= epsilon_threshold

        self.steps = 0
        self.reward_info = {}


        self.reset()
        #rendering configuration things
        self.fig = None
        self.ax = None
        self.robot = None
        self.joints = None
        self.targImg = None

    def update_psn(self):
        self._link_psns = self.kinematics.calc_link_positions()

    def _obs(self):
        x_end_effector = self.kinematics.calcEndEffector()

        th = self.kinematics.get_thetas()
        thvs = self.ang_v.copy()

        c = np.cos(th)
        s = np.sin(th)


        if self.obs_mode == 'complete':
            state = np.concatenate([x_end_effector, c, s, thvs, self.target])
        elif self.obs_mode == 'reduced':
            state = np.concatenate([th, thvs, self.target - x_end_effector]) #change

        return state

    def _generate_target(self):
        if self.target_generation == 'kinematic':
            #since we sample joint in space, kinematics are transforming samples
            #i.e. it is not a uniform distribution
            thetas = self.kinematics.get_thetas()
            #sample angles for each joint

            q_sample = np.zeros(self.NUM_JOINTS)
            for i in range(self.NUM_JOINTS):
                if i in self._control_joints:
                    low = self.joint_bounds.low[i]
                    high = self.joint_bounds.high[i]
                    bound = Bound(low if low > -np.inf else 0.0, high if high < np.inf else 2*np.pi)
                    q_s = np.random.uniform(bound.low, bound.high)
                    q_sample[i] = np.clip(q_s, bound.low, bound.high)

            self.kinematics.set_thetas(q_sample)
            self.target = self.kinematics.calcEndEffector().copy()
            self.kinematics.set_thetas(thetas)
        elif self.target_generation == 'half-sphere' or self.target_generation == 'cube':
            #could maybe do separately...
            self.target = np.array(self.target_sampler.sample())
        elif self.target_generation == "fixed":
            self.target = self.fixed_target
        else:
            raise Exception("Invalid target generation {}".format(self.target_generation))



    def _calc_reward(self, state, action, state_p):
        self.reward_info = {}
        if self.obs_mode == 'complete':
            x_t = state_p[-3:]
            x_e = state_p[0:3]

            x_prev = state[0:3]

            difference = x_t - x_e
        elif self.obs_mode == 'reduced':
            difference = state_p[-3:]
            x_prev = state[-3:]

        l2_dist = np.linalg.norm(difference, ord=2)

        self.reward_info['l2'] = -l2_dist

        self.reward_info['l1'] = -np.linalg.norm(difference, ord=1)

        self.reward_info['precision'] = np.exp( -l2_dist / self.precision_tau)

        #The idea is to penalize agent for being far away from target, the below supposedly helps to encourage moving to target faster
        self.reward_info['discrete-time-penalty'] = -1.0

        self.reward_info['keep-moving'] = np.linalg.norm(x_e - x_prev, 2)

        self.reward_info['action-norm'] = -np.linalg.norm(action, 2) ** 2 #squared action norm

        reward = 0.0
        for rew in self.reward_type:
            #exclude distance rewards if close to target
            if rew in ['discrete-time-penalty', 'keep-moving'] and l2_dist < self.epsilon_threshold:
                continue
                
            reward += self.reward_info[rew]

        return reward

    def step(self, a=None):
        if a is not None:
            self.ang_v = np.clip(a, -self._max_velocity, self._max_velocity)

        if self.boundary_mode == 'angular':
            low = self.joint_bounds.low
            high = self.joint_bounds.high
            current = self.kinematics.get_thetas()

            in_bound = self.check_boundary(low, high, current, self.ang_v, self.dt)
            self.ang_v = self.ang_v * in_bound
            #check all angles within boudns
        elif self.boundary_mode == 'position':
            print('check position')


        thetas = self.kinematics.get_thetas()
        self.kinematics.set_thetas(thetas + self.ang_v * self.dt)
        self.update_psn()

        self.steps += 1
        done = self.H <= self.steps

        state = self._obs()
        action = self.ang_v #not 100% sure this is right choice for this or not :/
        reward = self._calc_reward(self.prev_state, action, state)
        self.prev_state = state
        info  = self.reward_info.copy()
        return state, reward * self.dt, done, info


    def reset(self, thetas=np.array([0., 0.5, 0, -0.5, 0, 0.5, 0]),
            target = None):

        self.fixed_target = target if target is not None else self.fixed_target

        self.kinematics.set_thetas(thetas)
        self.update_psn()
        self._generate_target()

        self.prev_state = self._obs()
        self.steps = 0
        return self.prev_state

    def _initFig(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        bound = sum(self.link_lengths)
        x = bound
        y = bound
        z = bound

        self.ax.set(xlim=(-x, x), ylim=(-y, y), zlim=(-0, z))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])


    def joint_dim(self, i):
        return np.array([l[i] for l in self._link_psns])

    def render(self, mode='human'):

        if self.fig is None and self.ax is None:
            self._initFig()

        x = self.joint_dim(0)
        y = self.joint_dim(1)
        z = self.joint_dim(2)

        targ_x = [self.target[0]]
        targ_y = [self.target[1]]
        targ_z = [self.target[2]]
        if self.targImg is None:
            self.targImg = self.ax.scatter(targ_x, targ_y, targ_z)
        else:
            self.targImg._offsets3d = (targ_x, targ_y, targ_z)

        if self.joints is None:
            self.joints = self.ax.scatter(x, y, z)
        else:
            self.joints._offsets3d = (x, y, z)

        if self.robot is None:
            self.robot,  = self.ax.plot(x, y, z)
        else:
            self.robot.set_xdata(x)
            self.robot.set_ydata(y)
            self.robot.set_3d_properties(z)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return None

class JacobianWrapper(Wrapper):

    #TODO: ActionWrapper is probably...more accurate
    def __init__(self, env):

        assert isinstance(env, SimulatorKinovaGripper), "this wrapper only works for SimulatorKinovaGripper"

        env.action_space = spaces.Box(low=-env._max_velocity,
                high=env._max_velocity, shape=(len(env._control_joints) * 3,))

        super(JacobianWrapper, self).__init__(env)

        self._control_joints = env._control_joints

    def step(self, action):
        if action is not None:
            J = np.reshape(action, (3, len(self.env._control_joints)))

        iJ = np.linalg.pinv(J)

        x_end_effector = self.env.kinematics.calcEndEffector()
        diff = self.env.target - x_end_effector
        q_th = np.matmul(iJ, diff)

        return self.env.step(q_th)





class SimulatorKinovaGripperInverseJacobian(SimulatorKinovaGripper):
    def __init__(self,
                    dt=0.05,
                    max_velocity=None,
                    control_joints = list(range(7)),
                    joint_bounds=None,
                    H=100,
                    reward_type='l2',
                    precision_tau=0.01,
                    epsilon_threshold=0.1,
                    target_generation = 'kinematic',
                    radius_bound = [0.6, 1.07], #lower bound is psuedo arbitrary
                    phi_bound =[0.0, 1.0], #to be be more cone like shrink upper bound
                    theta_bound = [0.0, np.pi * 2],
                    fixed_target = np.array([0.6, 0.6, 0.6])
                    ):
        super(SimulatorKinovaGripperInverseJacobian, self).__init__(
                dt,
                max_velocity,
                control_joints,
                joint_bounds,
                H,
                reward_type,
                precision_tau,
                epsilon_threshold,
                target_generation,
                radius_bound, #lower bound is psuedo arbitrary
                phi_bound, #to be be more cone like shrink upper bound
                theta_bound,
                fixed_target)

        #TODO: I'm not quite sure what would be an actual good bound for this quite frankly...
        self.action_space = spaces.Box(low=-self._max_velocity
                , high=self._max_velocity, shape=(len(control_joints) * 3, ))

        self.J = np.zeros((len(control_joints), 3))


    def step(self, a=None):

        if a is not None:
            a = np.reshape(a, (len(self._control_joints), 3))

            self.J = a
        x_end_effector = self.kinematics.calcEndEffector()

        diff = self.target - x_end_effector
        #TODO this is actually iJ not J....
        q_th = np.matmul(self.J, diff)
        return super(SimulatorKinovaGripperInverseJacobian, self).step(q_th)



if __name__ == "__main__":
    env = SimulatorKinovaGripper(target_generation = 'half-sphere')
    i = 0
    while True:
        if i % 40:
            env.reset()
            i = 0
        else:
            i += 1
        env.render()
        act = np.ones(7) #np.random.randn(7)
        obs = env.step(act)


