import numpy as np
from environments.ExplicitKinematics import ExplicitKinematics

class DHKinematics(ExplicitKinematics):
    def __init__(self, ths_in_rads, using_gripper=True):
        super(DHKinematics, self).__init__(ths_in_rads, using_gripper)

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
    def T12(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.1564 + 0.1284), th=self.ths[0])

    def T23(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.0054 + 0.0064), th=self.ths[1] + np.pi)

    def T34(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.2104 + 0.2104), th=self.ths[2] + np.pi)

    def T45(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.0064 + 0.0064), th=self.ths[3] + np.pi)

    def T56(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=-(0.2084 + 0.1059), th=self.ths[4] + np.pi)

    def T67(self):
        return self.DHMatrix(alpha=np.pi / 2, a=0.0, d=0.0               , th=self.ths[5] + np.pi)

    def T7e(self):
        return self.DHMatrix(alpha=np.pi    , a=0.0, d=-(0.1059 + (0.0615 + self.gripper_offset)), th=self.ths[6] + np.pi)




