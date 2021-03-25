import jax.numpy as np
from jax import jacfwd, jit, grad
from visualservoing.policy_base import Policy
from numpy.linalg import lstsq #because i'm using jax as "numpy" in this

class DifferentialKinematics(object):
    #mostly a copy of DHKinematics from another module...
    # key thing is we  can differentiate with this one...although thats more cause of jax
    def __init__(self, using_gripper=True):
        
        #based on documents distance of gripper:
        self.gripper_offset = 0.120 if using_gripper else 0.0


    def _cosSin(self, th):
        return np.cos(th), np.sin(th)

    def DHMatrix(self, alpha, a, d, th):
        cth, sth = self._cosSin(th)
        ca, sa = self._cosSin(alpha)

        return np.array([
            [cth, -ca * sth, sa * sth, a * cth],
            [sth,  ca * cth,-sa * cth, a * sth],
            [  0,        sa,       ca,       d],
            [  0,         0,        0,       1]
            ])

    def TB1(self):
        return self.DHMatrix(alpha=np.pi    , a=0.0, d=0.0               , th=0.0)
    def T12(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.1564 + 0.1284), th=th)

    def T23(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.0054 + 0.0064), th=th + np.pi)

    def T34(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.2104 + 0.2104), th=th + np.pi)

    def T45(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.0064 + 0.0064), th=th + np.pi)

    def T56(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.2084 + 0.1059), th=th + np.pi)

    def T67(self, th):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=0.0               , th=th + np.pi)

    def T7e(self, th):
        return self.DHMatrix(alpha=np.pi    , a=0.0, d=-(0.1059 + (0.0615 + self.gripper_offset)), th=th + np.pi)

    def calc_pose_and_origin(self, th):
        c = 0.1
        pts =  np.array([
            [c, 0., 0., 0.],
            [0, c , 0., 0.],
            [0, 0., c , 0.],
            [1., 1., 1., 1.],
        ])
        T02 = np.matmul(self.TB1(), self.T12(th[0]))
        T03 = np.matmul(T02, self.T23(th[1]))
        T04 = np.matmul(T03, self.T34(th[2]))
        T05 = np.matmul(T04, self.T45(th[3]))
        T06 = np.matmul(T05, self.T56(th[4]))
        T07 = np.matmul(T06, self.T67(th[5]))
        T0e = np.matmul(T07, self.T7e(th[6]))


        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :].flatten('F')

    def calc_pose(self, th):
        c = 0.1
        pts =  np.array([
            [c, 0., 0.],
            [0, c , 0.],
            [0, 0., c ],
            [1, 1., 1.],
        ])
        T02 = np.matmul(self.TB1(), self.T12(th[0]))
        T03 = np.matmul(T02, self.T23(th[1]))
        T04 = np.matmul(T03, self.T34(th[2]))
        T05 = np.matmul(T04, self.T45(th[3]))
        T06 = np.matmul(T05, self.T56(th[4]))
        T07 = np.matmul(T06, self.T67(th[5]))
        T0e = np.matmul(T07, self.T7e(th[6]))
        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :].flatten('F')

    def calc_min_pose_and_origin(self, th):
        c = 0.1
        pts =  np.array([
            [c, 0., 0.],
            [0, c , 0.],
            [0, 0., 0.],
            [1, 1., 1.],
        ])
        T02 = np.matmul(self.TB1(), self.T12(th[0]))
        T03 = np.matmul(T02, self.T23(th[1]))
        T04 = np.matmul(T03, self.T34(th[2]))
        T05 = np.matmul(T04, self.T45(th[3]))
        T06 = np.matmul(T05, self.T56(th[4]))
        T07 = np.matmul(T06, self.T67(th[5]))
        T0e = np.matmul(T07, self.T7e(th[6]))
        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :].flatten('F')

    def calcEndEffector(self, th):
        p = np.array([0., 0., 0., 1.])
        T02 = np.matmul(self.TB1(), self.T12(th[0]))
        T03 = np.matmul(T02, self.T23(th[1]))
        T04 = np.matmul(T03, self.T34(th[2]))
        T05 = np.matmul(T04, self.T45(th[3]))
        T06 = np.matmul(T05, self.T56(th[4]))
        T07 = np.matmul(T06, self.T67(th[5]))
        T0e = np.matmul(T07, self.T7e(th[6]))

        psn = np.matmul(T0e, p)
        return psn[0:3]

def norm_loss(psn, targ, n=2):
    #L1 loss is less effective it seems like than l2 for true system 
    return np.linalg.norm(targ - psn, n)


   
class InverseJacobian(Policy):
    def __init__(self, gain, state_extractor, direct_solve=False):
        super(InverseJacobian, self).__init__( gain, state_extractor)

        self.kinematics = DifferentialKinematics()
        self.jacobian = jit(jacfwd(self.kinematics.calcEndEffector))
        self.J = None

        self.direct_solve= direct_solve
        #the idea was minimize norm loss as we would do in regular supervised learning
        #i've found this to be kinda slow....so just like fyi
        self.grad = jit(grad(lambda q, targ: norm_loss(self.kinematics.calcEndEffector(q),
                                                    targ), argnums=0))

    def save(self, pth='inverse_jacobian.pth'):
        print("nothing worth saving for InverseJacobian Controller")

    def load(self, pth='inverse_jacobian.npy'):
        print("nothing worth saving for InverseJacobian Controllers")

    def reset(self):
        #nothing to reset
        _ = 0
    def learn(self, gym, external_data=None):
        #nothing to learn...we assume underlying dynamics already known
        _ = 0

    def act(self, obs):
        #TODO: just like...an FYI:
        #for multi point environment...this doesn't bug out but probably should....
        ths = self.state_extractor.get_angles( obs)
        psn, trg = self.state_extractor.get_position_and_target( obs)
        
        J = self.jacobian(ths)
        self.J = J

        #if np.abs(np.linalg.det(J)) < 1e-10:
        #    print("links close to singular")
        #    print(J)
        #    J = J + 1e-10
        iJ = np.linalg.pinv(J)

        #calculate action
        if self.direct_solve:

            action = -1.0 * self.gain * self.grad(ths, trg)
        else:
            x_dot = trg - psn
            th_dot = np.matmul(iJ, x_dot)
            action = (self.gain * th_dot)

        return action

"""
    Nearly Identifical to above class, just for 4 points instead of just 1
"""
class MultiPointInverseJacobian(InverseJacobian):

    def __init__(self, gain, state_extractor, pts_config="pose_and_origin"):
        super(MultiPointInverseJacobian, self).__init__( gain, state_extractor)
        
        self.kinematics = DifferentialKinematics()
        self.jacobian = None
        if pts_config == "pose_and_origin":
            self.jacobian = jit(jacfwd(self.kinematics.calc_pose_and_origin))
        elif pts_config ==  "pose":
            self.jacobian = jit(jacfwd(self.kinematics.calc_pose))
        elif pts_config == "min_pose_and_origin":
            self.jacobian = jit(jacfwd(self.kinematics.calc_min_pose_and_origin))


        #self.jacobian = jit(jacfwd(self.kinematics.calc_pose_and_origin))
        self.J = None

"""
    2DOF Inverse Jacobian
"""

class TwoDOFInverseJacobian(InverseJacobian):
    def __init__(self, L1, L2, gain = 1.0, 
            n_upd=1.0, sample_upd=False, temp=0.1, stop_move=False, stop_before_n=0,
            homography=None):
        #target: position we want to solve for
        #lr: learning rate i.e. how much of gradient to return as action
        self.L1 = L1
        self.L2 = L2
        self.gain = gain 

        #parameters for action frequency 
        #update every nth calcualation
        self.n_upd = int(n_upd)
        self.since_upd = 0
        self.prev_act = None

        #to send 0 before updating speed
        assert stop_before_n < n_upd, "stop_before_n must be less than frequency of n updates"
        self.stop_move = stop_move
        self.stop_before_n = int(stop_before_n)

        #sample actions based on distance
        self.sample_upd = sample_upd
        self.temp = temp

        #storing previous action taken
        self.last_act = np.array([0.0, 0.0])

        #homography if not none, assume to transform
        self.homography = homography

    def sample_act(self, x_dot):
        dist = np.linalg.norm(x_dot)
        prob = pow(np.e, -dist/ self.temp)
        #if 1 sample action
        return 1 == np.random.binomial(1, prob)

    def transform(self, a, H):
        b = H.dot ( np.concatenate((a.T,np.ones((1,a.shape[0]))),axis=0)).T
        b /= b[:,-1:]
        return b[:,:-1]

    def act(self, obs):
        #given obs from environment, calculate 2D solution

        #Calculate Jacobian matrix 
        c_th1 = obs[2]
        c_th2 = obs[3]
        s_th1 = obs[4]
        s_th2 = obs[5]

        th1 = np.arctan2(s_th1, c_th1)
        th2 = np.arctan2(s_th2, c_th2)

        #Calculate Values in Jacobian
        L1 = self.L1
        L2 = self.L2
        C1 = np.cos(th1)
        S1 = np.sin(th1)
        C12 = np.cos(th1 + th2)
        S12 = np.sin(th1 + th2)

        #Actual Jacobian matrix
        J = np.array([
            [-L1*S1 - L2*S12, -L2*S12],
            [ L1*C1 + L2*C12,  L2*C12],
        ])
        
        self.J = J
        if np.abs(np.linalg.det(J)) < 1e-10:
            print("links close to singular")
            print(J)
            J = J + 1e-10


        iJ = np.linalg.inv(J)
        
        #Get dx i.e. distance between target and current position
        psn = np.array(obs[0:2])
        trg = np.array(obs[8:10])

        if self.homography is not None:
            psn = self.transform(psn.reshape(1,2)*np.array([[640,320]]), self.homography ).reshape(2)
            trg = self.transform(trg.reshape(1,2)*np.array([[640,320]]), self.homography ).reshape(2)
            
 
        #calculate action
        x_dot = trg - psn
        th_dot = np.matmul(iJ, x_dot)
        action = (self.gain * th_dot)

        # determine action behavior
        if self.sample_upd:
            if self.prev_act is None:
                self.prev_act = action
                return action
            elif self.sample_act(x_dot):
                self.prev_act = action
                return action
            else:
                action = self.prev_act
                return action
        else:
            self.since_upd = self.since_upd % self.n_upd

            if self.stop_move and self.since_upd == (self.n_upd - self.stop_before_n):
                action = np.array([0.0, 0.0])
            elif self.since_upd != 0:
                action = self.prev_act

            #update updates we've done and keep track of previous action
            self.since_upd += 1
            self.prev_act = action
            return action


