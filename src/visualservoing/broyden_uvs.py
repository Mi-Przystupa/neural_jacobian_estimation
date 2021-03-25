#!/usr/bin/env python

import numpy as np
from visualservoing.policy_base import Policy

class Broyden:
    def __init__(self, init_th, init_im_f, alpha=1.0, dth_eps=0.1, n_steps=10, jac_diff_thresh=30.0):
        self.alpha = alpha
        self.dth_eps = dth_eps 
        self.n_steps = n_steps
        self.jac_diff_thresh = jac_diff_thresh
        self.resetState(init_th, init_im_f)
        print("we only check if change in angle is sufficients")

    def makeColumn(self, vec):
        if len(vec.shape) < 2:
            vec = np.reshape(vec, (vec.shape[0], -1))
        return vec

    def resetState(self, th, im_f):
        self.steps = int(0)
        self.init_th = th
        self.init_im_f = im_f

    def calculateUpdate(self, J_k, d_ths, d_e):
        #2x2 matrix matmul with 2x1 column vector ==> 2x1 output
        JkDths = np.matmul(J_k, d_ths)
        #difference of 2x1 column vectors
        diff = d_e - JkDths
        #2x1 matrix matmul with 1 x 2 matrix ==> 2x2 matrix
        num = np.matmul(diff, d_ths.T)

        #dot product of a column vector d_ths
        denom = np.matmul(d_ths.T, d_ths)

        update = num / denom
        J_k_p = J_k + self.alpha * update 
        return J_k_p

    def conditionsMet(self, d_ths):
        #if the denominator is sufficiently large or we've reached some x number of steps 
        denom_magn = np.matmul(d_ths.T, d_ths)
        return denom_magn >= self.dth_eps or self.steps >= self.n_steps

    def viableUpdate(self, J_k, J_k_p):
        #given J_k and J_k_p determine if norm between them is sufficiently small
        difference = np.linalg.norm(J_k_p - J_k)
        make_update = difference >= self.jac_diff_thresh

        return True#make_update

    def init_conditions_set(self):
        return self.init_th is not None and self.init_im_f is not None

    def updateState(self):
        #tick number of steps since last polled values
        self.steps += 1

    def attempt_update(self, J_k, th, im_f):
        """
        J_k: 2x2 matrix 
        im_f: image feature, if shape is (2,) will be reshaped to (2, 1)
        th: angle, if shape is (2,) will be reshaped to (2,1)
        """
        if not self.init_conditions_set():
            self.resetState(th, im_f)
            return J_k

        #if e = trg - psn ==> e' - e = psn - psn'
        d_e = self.makeColumn(im_f - self.init_im_f)#
        d_ths = self.makeColumn(th - self.init_th) 

        if self.conditionsMet(d_ths):
            J_k_p = self.calculateUpdate(J_k, d_ths, d_e)
            self.resetState(th, im_f)
            return J_k_p if self.viableUpdate(J_k, J_k_p) else J_k
        else: 
            self.updateState()
            return J_k

class BFGS(Broyden):
    def __init__(self, init_th, init_im_f, alpha=1.0, dth_eps=0.1, n_steps=10, jac_diff_thresh=30.0):
        super(BFGS, self).__init__(init_th, init_im_f, alpha= alpha, dth_eps= dth_eps, n_steps=n_steps, jac_diff_thresh=jac_diff_thresh)
        #BFGS only makes sense with square matrices so if number of features == number of actuators this could work

    def calculateUpdate(self, J_k, d_ths, d_e):
        [m, _] = d_e.shape
        [n, _] = d_ths.shape
        
        #I don't think...this will actually make it work, just like...fyi this is like extra super hacky....
        #if the matrix isn't square... we will MAKE it be square and then just take the rows and columns we want
        if m < n:
            #number of features  is less than number of actuators so pad features 
            d = n - m
            pad = np.random.rand(d, 1)
            d_e = np.concatenate([d_e, pad])
            pad = np.random.rand(d, n)
            J_k = np.concatenate((J_k, pad)) 

        elif m > n:
            d = m - n
            pad = np.random.rand(d, 1)
            d_ths = np.concatenate([d_ths, pad])
            pad = np.random.rand(m, d)
            J_k = np.concatenate((J_k, pad), axis=1) 

        #alpha term
        num = np.matmul(d_e, d_e.T)
        denom = np.matmul(d_ths.T, d_e)

        alpha = num / denom

        #beta term
        Jk_dq = np.matmul(J_k, d_ths)
        num = np.matmul(Jk_dq, Jk_dq.T)
        denom = np.matmul(d_ths.T, Jk_dq)

        beta = num / denom
        
        update = alpha - beta

        new = J_k + self.alpha * update
        return new[:m, :n]


    def viableUpdate(self, J_k, J_k_p):
        #always take the update
        return True


class UncalibratedVisuoServoing(Policy):
    def __init__(self, gain, n_updates, num_actuators=7, alpha=0.1, test_act=0.5,
            calib_act_val=0.5, use_broyden=False, use_BFGS=False, n_step_broyden=np.inf, dth_eps=0.01,
            state_extractor=None):

        super(UncalibratedVisuoServoing, self).__init__(gain=gain, state_extractor=state_extractor)

        self.num_actuators = num_actuators

        self.init_im_feats = None
        self.end_im_feats = None
        self.init_q = None
        self.end_q = None

        
        self.calib_act_val = calib_act_val
        self.J = np.zeros((num_actuators, self.state_extractor.get_num_features()))


        #stuff for secant updates

        self.broyden = None
        if use_broyden:
            self.broyden = Broyden(None, None, alpha=alpha, dth_eps=dth_eps, n_steps=n_step_broyden)
        elif use_BFGS:
            #warning: only makes sense if num_features == num_actuators
            self.broyden = BFGS(None, None, alpha = alpha, dth_eps= dth_eps, n_steps= n_step_broyden)

        self.J_k = None

        #stuff related to running and setting initial jacobian
        self.n_updates = int(n_updates)
        self.updates = 0.0
        self.test_act = test_act
        
    def reset(self):
        self.prev_ths = None
        self.J_k = None
        if self.broyden is not None:
            self.broyden.resetState(None, None)

    def learn(self, gym, external_data=None):
        self.initJacobian(gym)

    def load(self, pth='broyden_init_jacobian.pth'):
        features = np.load(pth).item()
        self.init_im_feats = features['init_im_feats']
        self.end_im_feats = features['end_im_feats']
        self.init_q = features['init_q']
        self.end_q = features['end_q']

    def save(self, pth='broyden_init_jacobian.pth'):
        features = {
                'init_im_feats': self.init_im_feats,
                'end_im_feats': self.end_im_feats,
                'init_q': self.init_q,
                'end_q': self.end_q 
                }

        with open(pth, 'wb') as f:
            np.save(f, features)
    
    def initJacobian(self, gym):
        #interact with environment to first get approximate Jacobian

        #our notion of image feature is really only where the position of the tracker is.
        #I suppose mathematically we are then definine e = psn - target
        # e' - e = (psn' - target) - (psn - target) = psn' - psn

        def calibrateJoint(gym, joint):
            obs = gym.reset()
            init_angs = self.state_extractor.get_angles(obs)
            init_posn = self.state_extractor.get_position(obs)

            act = np.zeros(self.num_actuators)
            act[joint] = self.calib_act_val if (joint % 2) == 0 else -1.0 * self.calib_act_val

            for _ in range(self.n_updates):
                output = gym.step(act)
                obs = output[0]
                psn = self.state_extractor.get_position(obs)

                delta = np.abs(psn - init_posn)

                finished = np.all(delta >= 10.0)
                if finished:
                    break
            #stop movement, is for kinova
            gym.step(np.zeros(self.num_actuators))
            end_angs = self.state_extractor.get_angles(obs)
            end_psn = self.state_extractor.get_position(obs)

            return init_angs[joint], end_angs[joint], init_posn,  end_psn
        self.init_im_feats = np.zeros((self.num_actuators, self.state_extractor.get_num_features()))
        self.end_im_feats = np.zeros((self.num_actuators, self.state_extractor.get_num_features()))
        self.init_q = np.zeros((self.num_actuators, 1))
        self.end_q = np.zeros((self.num_actuators, 1))

        #move for 1st angle
        for i in range(self.num_actuators):
            init_qs, end_qs, init_psn,  end_psn = calibrateJoint(gym, i)
            self.init_im_feats[i] = init_psn
            self.end_im_feats[i] = end_psn
            self.init_q[i] = init_qs
            self.end_q[i] = end_qs

 
    def finite_deriv(self, fn_init, fn_end, dt_init, dt_end):
        return (fn_end - fn_init) / (dt_end - dt_init)

    def Jacobian(self):
        jac = self.finite_deriv(self.init_im_feats, self.end_im_feats, self.init_q, self.end_q)
        return jac.T

    

    def act(self, obs):
        ths = self.state_extractor.get_angles(obs)
        psn, trg = self.state_extractor.get_position_and_target(obs)
        
        #error
        #for some reason below works
        #although our jacobian I think mathematically is using e = psn - trg (just look at initJacobian to see what I'm getting at)
        x_dot = trg - psn #documents would suggest should be psn - trg 

        # loop to set image features for 1st column
        if self.J_k is None:
            self.J_k = self.Jacobian()

        if self.broyden is not None:
            curr_ths = ths 
            self.J_k = self.broyden.attempt_update(self.J_k, curr_ths, psn)
            

        J = self.J_k
        self.J = J
        #d = np.linalg.det(J)
        
        iJ = np.linalg.pinv(J)
        th_dot = np.matmul(iJ, x_dot)

        action = self.gain * th_dot 
        return action

