import numpy as np

class ExplicitKinematics(object):
    def __init__(self, ths_in_rads, using_gripper=True):
        self.ths = ths_in_rads 
        #based on documents distance of gripper:
        self.gripper_offset = 0.120 if using_gripper else 0.0


    def _cosSin(self, th):
        return np.cos(th), np.sin(th)

    def TB1(self):
        c, s = self._cosSin(self.ths[0])
        return np.array([
            [ c,-s, 0, 0],
            [-s,-c, 0, 0],
            [ 0, 0,-1, 0.1564],
            [ 0, 0, 0, 1]
        ])

    def T12(self):
        c, s= self._cosSin(self.ths[1])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0,-1, 0.0054],
            [ s, c, 0,-0.1284],
            [ 0, 0, 0, 1]
        ])

    def T23(self):
        c, s= self._cosSin(self.ths[2])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0, 1,-0.2104],
            [-s,-c, 0,-0.0064],
            [ 0, 0, 0, 1]
        ])

    def T34(self):
        c, s= self._cosSin(self.ths[3])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0,-1,-0.0064],
            [ s, c, 0,-0.2104],
            [ 0, 0, 0, 1]
        ])


    def T45(self):
        c, s= self._cosSin(self.ths[4])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0, 1,-0.2084],
            [-s,-c, 0,-0.0064],
            [ 0, 0, 0, 1]
        ])

    def T56(self):
        c, s= self._cosSin(self.ths[5])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0,-1, 0],
            [ s, c, 0,-0.1059],
            [ 0, 0, 0, 1]
        ])

    def T67(self):
        c, s = self._cosSin(self.ths[6])
        return np.array([
            [ c,-s, 0, 0],
            [ 0, 0, 1,-0.1059],
            [-s,-c, 0, 0],
            [ 0, 0, 0, 1]
        ])

    def T7e(self):
        
        return np.array([
            [ 1, 0, 0, 0],
            [ 0,-1, 0, 0],
            [ 0, 0,-1,-0.0615 - self.gripper_offset],
            [ 0, 0, 0, 1]
        ])

    def set_thetas(self, ths):
        self.ths = ths

    def get_thetas(self):
        return self.ths

    def calc_link_positions(self):
        p = [0., 0., 0., 1.]

        T02 = np.matmul(self.TB1(), self.T12())
        T03 = np.matmul(T02, self.T23())
        T04 = np.matmul(T03, self.T34())
        T05 = np.matmul(T04, self.T45())
        T06 = np.matmul(T05, self.T56())
        T07 = np.matmul(T06, self.T67())
        T0e = np.matmul(T07, self.T7e())
    
        transforms = [T02, T03, T04, T05, T06, T07, T0e]
        links = [None] * len(transforms) 
        for i, t in enumerate(transforms):
            links[i] = np.matmul(t, p)[0:3]

        return links

    def calc_pose_and_origin(self):
        #remove 1 feature?
        pts = np.eye(4) *.1
        pts[3,:] = 1.0
        T02 = np.matmul(self.TB1(), self.T12())
        T03 = np.matmul(T02, self.T23())
        T04 = np.matmul(T03, self.T34())
        T05 = np.matmul(T04, self.T45())
        T06 = np.matmul(T05, self.T56())
        T07 = np.matmul(T06, self.T67())
        T0e = np.matmul(T07, self.T7e())

        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :]

    def calc_pose(self):
        #remove 1 feature?
        pts = np.eye(4) *.1
        pts[3,:] = 1.0
        pts = pts[:,0:3]

        T02 = np.matmul(self.TB1(), self.T12())
        T03 = np.matmul(T02, self.T23())
        T04 = np.matmul(T03, self.T34())
        T05 = np.matmul(T04, self.T45())
        T06 = np.matmul(T05, self.T56())
        T07 = np.matmul(T06, self.T67())
        T0e = np.matmul(T07, self.T7e())

        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :]

    def calc_min_pos_origin(self):
        #remove 1 feature?
        pts = np.eye(4) *.1
        pts[3,:] = 1.0
        pts = pts[:,[0,1,3]]

        T02 = np.matmul(self.TB1(), self.T12())
        T03 = np.matmul(T02, self.T23())
        T04 = np.matmul(T03, self.T34())
        T05 = np.matmul(T04, self.T45())
        T06 = np.matmul(T05, self.T56())
        T07 = np.matmul(T06, self.T67())
        T0e = np.matmul(T07, self.T7e())

        psn = np.matmul(T0e, pts)
        #last row is homogenous translation component
        return psn[0:3, :]



    def calcEndEffector(self):
        p = [0., 0., 0., 1.]
        T02 = np.matmul(self.TB1(), self.T12())
        T03 = np.matmul(T02, self.T23())
        T04 = np.matmul(T03, self.T34())
        T05 = np.matmul(T04, self.T45())
        T06 = np.matmul(T05, self.T56())
        T07 = np.matmul(T06, self.T67())
        T0e = np.matmul(T07, self.T7e())

        psn = np.matmul(T0e, p)
        return psn[0:3]


   
