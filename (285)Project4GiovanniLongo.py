from calendar import c
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import copy
from textwrap import wrap

# hard set a global variables for all cases
N = 30
epsilon = 1/1_000_000

class Parabolic():
    '''
    A class for the parabolic problem.
    '''
    def __init__(self, u_a, u_b, u_0, c, d, f, BTL, BTR):
        self.x = create_x(N)

        # our boundary values at x = a and x = b
        self.ua = u_a
        self.ub = u_b

        # our defined initial vector u at t = 0
        self.u0 = u_0
        self.c = c
        self.d = d
        self.f = f

        # the boundary type for the beginning point and end point
        self.BL = BTL
        self.BR = BTR
        
        self.U0 = []

    def c_x(self, xmid):
        '''
        A function that returns the correct vale of d(x).
        '''
        if self.c == "x":
            return xmid
        elif self.c == "x2":
            return (xmid) ** 2
        
        return self.c


    # create separate function for d(x) in case c(x) != d(x)
    def d_x(self, xmid):
        '''
        A function that returns the correct vale of d(x).
        '''
        if self.d == "x":
            return xmid
        elif self.d == "x2":
            return (xmid) ** 2
        
        return self.d

    
    def f_x(self, xmid, t):
        '''
        A function that gives the correct value of f(x).
        '''
        # we have an extra variable 'xmid' in case f(x, t) = u(x, t) also had
        # an x dependence
        if self.f == "f":
            return t

        return 0


class simTime():
    '''
    Object to control the time step information and reject/accept time steps.
    '''
    def __init__(self, dt, tol, grow, shrink, dtmin, dtmax, endTime):
        self.time = 0
        self.dt = dt
        self.tol = tol
        self.grow_fac = grow
        self.shrink_fac = shrink
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.endTime = endTime
        self.stepsSinceRejection = 0
        self.stepsRejected = 0
        self.stepsAccepted = 0
        self.time_lst = []
        self.dt_lst = []

    def checker(self, e_i):
        '''
        Checks if a given step is acceptable or must have an altered dt step.
        '''
        if e_i > self.tol and self.dt > self.dtmin:
            self.dt = max(self.dtmin, self.dt / 2)
            self.stepsSinceRejection = 0
            self.stepsRejected += 1
            return False
            
        else:
            self.stepsAccepted += 1
            self.stepsSinceRejection += 1
            self.time_lst.append(self.time)
            self.dt_lst.append(self.dt)
            self.time += self.dt
            if e_i > 0.75 * self.tol:
                self.dt = self.dt * self.shrink_fac
            elif e_i < 0.25 * self.tol:
                if self.stepsSinceRejection > 1:
                    self.dt = self.dt * self.grow_fac
            self.dt = min(self.dtmax, max(self.dt, self.dtmin))
            if self.time + self.dt > self.endTime:
                self.dt = self.endTime - self.time
            elif self.time + (2 * self.dt) > self.endTime:
                self.dt = (self.endTime - self.time) / 2
            return True


    
def resid(U, dU, PP, t, dt):
    '''
    Calculates the right hand side of the dU equation.
    '''
    r = np.zeros(N + 1)
    for i in range(0, N):
        dx = PP.x[i + 1] - PP.x[i]
        xmid = (PP.x[i +1] + PP.x[i])/ 2
        r[i] += (PP.c_x(xmid) * (dx / dt) * ((dU[i] / 3) + (dU[i + 1] / 6))) + \
            ((PP.d_x(xmid) / dx) * (U[i] - U[i+1])) - \
                ((PP.f_x(xmid, t + dt)) * (dx / 2))

        r[i + 1] += (PP.c_x(xmid) * (dx / dt) * ((dU[i] / 6) + (dU[i + 1] / 3))) + \
            ((PP.d_x(xmid) / dx) * (U[i + 1] - U[i])) - \
                ((PP.f_x(xmid, t + dt)) * (dx / 2))

    # check left boundary condition
    if PP.BL == "D":
        r[0] = U[0] - PP.ua
    elif PP.BL == "N":
        r[0] -= PP.ua

    # check right boundary condition
    if PP.BR == "D":
        r[-1] = U[-1] - PP.ub
    elif PP.BR == "N":
        r[-1] += PP.ub

    return r    


def step(PP, U, t, dt):
    '''
    Goes forward a single step to find the right-hand vector needed for
    the next value of dU.
    '''
    r = resid(U, np.zeros(N + 1), PP, t, dt)
    c = create_C(U, PP, t, dt)
    
    return solve_banded((1,1), c, -r)


def advance(PP, SimT):
    '''
    Advances to a certain time step to find the corresponding approximation for U.
    '''
    if SimT.endTime == 0:
        return PP.u0

    U = copy.deepcopy(PP.u0)
    while SimT.time < SimT.endTime:
        S = U + step(PP, U, SimT.time, SimT.dt)
        D_first = U + step(PP, U, SimT.time, SimT.dt / 2)
        D = U + step(PP, D_first, SimT.time, SimT.dt / 2)
        # find the norm max of S and D
        e_i = abs(max(S-D))

        if SimT.checker(e_i):
            U = S
            PP.U0.append(U[0])

    return U


def create_x(N):
    '''
    Creates the vector x based off the function p(x).
    '''
    x = np.zeros(N + 1)
    for i in range(0, N + 1):
        y = i / N
        x[i] = y + (0.9 * y * (1-y))

    return x


def create_C(U, PP, t, dt):
    '''
    Creates the Jacobian matrix needed for tridiaganoal calculations.
    '''
    Rbase = resid(U, np.zeros(N + 1), PP, t, dt)
    c = np.zeros((3, N + 1))
    for i in range(0, 3):
        dU = np.zeros(N + 1)
        for j in range(i, N + 1, 3):
            dU[j] += epsilon
        R = (resid(U + dU, dU, PP, t, dt) - Rbase) / epsilon
        for k in range(i, N + 1, 3):
            c[0, k] = R[k - 1]
            c[1, k] = R[k]
            if k != N:
                c[2, k] = R[k + 1]

    return c


def plotter(x_lst, U_lst, t_lst, title, ylabel, xlabel="x-values"):
    '''
    Takes in a list of the desired vectors we want to plot and returns
	a plot with all plotted functions. Also takes in a list of names
	for the respective vectors to plot.
    '''
    for i in range(0, len(U_lst)):
        plt.plot(x_lst, U_lst[i], label=t_lst[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('\n'.join(wrap(title,60)))
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # Case 0
    u_0 = np.zeros(N + 1)
    for i in range (0, N + 1):
        u_0[i] = i / N

    case0 = Parabolic(1, 1, u_0, 1, 1, 0, "N", "N")

    SimT00 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 0)
    U_00 = advance(case0, SimT00)

    SimT01 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_01 = advance(case0, SimT01)

    SimT02 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_02 = advance(case0, SimT02)

    SimT03 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_03 = advance(case0, SimT03)

    SimT04 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1)
    U_04 = advance(case0, SimT04)

    plotter(np.array(SimT04.time_lst), [np.log10(np.array(SimT04.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 0.", "dt", "time")

    plotter(case0.x, [U_00, U_01, U_02, U_03, U_04], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1"], \
            "Double Neumann approximation with c=d=1, u_0=x, and f=0.", \
                "U(x)")

    
    # Case 1
    u_0 = np.zeros(N + 1)

    case1 = Parabolic(0, 1, u_0, 1, 1, 0, "N", "D")

    SimT10 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 0)
    U_10 = advance(case1, SimT10)

    SimT11 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_11 = advance(case1, SimT11)

    SimT12 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_12 = advance(case1, SimT12)

    SimT13 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_13 = advance(case1, SimT13)

    SimT14 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1)
    U_14 = advance(case1, SimT14)

    case1.U0 = []
    
    SimT15 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 2)
    U_15 = advance(case1, SimT15)

    plotter(np.array(SimT15.time_lst), [np.log10(np.array(SimT15.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 1.", "dt", "time")

    plotter(SimT15.time_lst, [case1.U0], \
        ["time vs. U[0]"], "Time vs. U[0] plot for Case 1.", "U[0]", "time")

    plotter(case1.x, [U_10, U_11, U_12, U_13, U_14, U_15], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1", "t=2"], \
            "Left Neumann and Right Dirichilet approximation with c=d=1, u_0=0, and f=0.", \
                "U(x)")


    # Case 2
    u_0 = np.zeros(N + 1)

    case2 = Parabolic(0, 1, u_0, "x", "x", 0, "N", "D")

    SimT20 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 0)
    U_20 = advance(case2, SimT20)

    SimT21 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_21 = advance(case2, SimT21)

    SimT22 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_22 = advance(case2, SimT22)

    SimT23 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_23 = advance(case2, SimT23)

    SimT24 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1)
    U_24 = advance(case2, SimT24)

    case2.U0 = []
    
    SimT25 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 2)
    U_25 = advance(case2, SimT25)

    plotter(np.array(SimT25.time_lst), [np.log10(np.array(SimT25.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 2.", "dt", "time")

    plotter(SimT25.time_lst, [case2.U0], \
        ["time vs. U[0]"], "Time vs. U[0] plot for Case 2.", "U[0]", "time")

    plotter(case1.x, [U_20, U_21, U_22, U_23, U_24, U_25], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1", "t=2"], \
            "Left Neumann and Right Dirichilet approximation with c=d=x, u_0=0, and f=0.", \
                "U(x)")


    # Case 3
    u_0 = np.zeros(N + 1)

    case3 = Parabolic(0, 1, u_0, "x2", "x2", 0, "N", "D")
    
    SimT30 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 0)
    U_30 = advance(case3, SimT30)

    SimT31 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_31 = advance(case3, SimT31)

    SimT32 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_32 = advance(case3, SimT32)

    SimT33 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_33 = advance(case3, SimT33)

    SimT34 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1)
    U_34 = advance(case3, SimT34)

    case3.U0 = []
    
    SimT35 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 2)
    U_35 = advance(case3, SimT35)

    plotter(np.array(SimT35.time_lst), [np.log10(np.array(SimT35.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 3.", "dt", "time")

    plotter(SimT35.time_lst, [case3.U0], \
        ["time vs. U[0]"], "Time vs. U[0] plot for Case 3.", "U[0]", "time")

    plotter(case3.x, [U_30, U_31, U_32, U_33, U_34, U_35], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1", "t=2"], \
            "Left Neumann and Right Dirichilet approximation with c=d=x^2, u_0=0, and f=0.", \
                "U(x)")
    

    # Case 4
    u_0 = np.ones(N + 1)

    case4 = Parabolic(0, 0, u_0, 1, 1, "f", "N", "N")
    
    SimT40 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 0)
    U_40 = advance(case4, SimT40)

    SimT41 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_41 = advance(case4, SimT41)

    SimT42 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_42 = advance(case4, SimT42)

    SimT43 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_43 = advance(case4, SimT43)

    SimT44 = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 1)
    U_44 = advance(case4, SimT44)

    plotter(np.array(SimT44.time_lst), [np.log10(np.array(SimT44.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 4.", "dt", "time")

    plotter(case4.x, [U_40, U_41, U_42, U_43, U_44], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1"], \
            "Double Neumann approximation with c=d=1, u_0=1, and f=t.", \
                "U(x)")
    

    # Case 5
    u_0 = np.zeros(N + 1)

    case5 = Parabolic(0, 1, u_0, "x", "x", 0, "N", "D")

    SimT50 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 0)
    U_50 = advance(case5, SimT50)

    SimT51 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 1/20)
    U_51 = advance(case5, SimT51)

    SimT52 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 1/10)
    U_52 = advance(case5, SimT52)

    SimT53 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 1/5)
    U_53 = advance(case5, SimT53)

    SimT54 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 1)
    U_54 = advance(case5, SimT54)

    case5.U0 = []
    
    SimT55 = simTime(0.0001, 0.001, 1.25, 0.8, 0.0001, 0.1, 2)
    U_55 = advance(case5, SimT55)

    plotter(np.array(SimT55.time_lst), [np.log10(np.array(SimT55.dt_lst))], \
        ["time vs. dt"], "Time vs. dt plot for Case 5.", "dt", "time")

    plotter(SimT55.time_lst, [case5.U0], \
        ["time vs. U[0]"], "Time vs. U[0] plot for Case 5.", "U[0]", "time")

    plotter(case1.x, [U_50, U_51, U_52, U_53, U_54, U_55], \
        ["t=0", "t=0.05", "t=0.1", "t=0.2", "t=1", "t=2"], \
            "Case 2 with dt tolerance set to 0.001.", \
                "U(x)")