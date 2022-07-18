from calendar import c
from inspect import stack
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import copy
from textwrap import wrap
from numba import njit

# hard set a global variables for all cases
N = 100
epsilon = 1/1_000_000

class Problem_Def():
    '''
    A class for 1-D evolution equation.
    '''
    def __init__(self, u_b, d, z, q, hbw, vars, case):
        self.x = np.arange(0, 10 + 1/10, 1/10)

        # our boundary values at x = a and x = b
        self.ua = u_a(0)
        self.ub = u_b

        # our defined initial vector u at t = 0
        self.u = np.zeros((vars, N + 1))
        self.u[0] = z
        self.u[1] = q

        self.d = d
        self.vars = vars
        self.f = 0
        self.hbw = hbw

        # the boundary type for the beginning point and end point
        
        self.U0 = []

        self.case = case


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


def f_x(q, z):
    '''
    Function which returns the value of f for the given z and q.
    '''
    if z < 0 or q < 0:
        return 0

    return q * z


def u_a(t):
    '''
    Function that handles the boundary condition for z(0,t).
    '''
    return 1 - np.exp(-5 * t)


def create_p():
    '''
    Creates the initial p vector for example 2.
    '''
    p = np.zeros(N + 1)
    for i in range(0, N + 1):
        x = (10 / N) * i
        if x <= 1:
            p[i] = 2 * ((1-x**2)**3)
        else:
            p[i] = 0

    return p


def interwoven(R, rows, columns):
    '''
    Transforms a stacked matrix into its interwoven form.
    '''
    RI = np.zeros(rows * columns)
    for j in range(0, columns):
        for i in range(0, rows):
           RI[(j * rows) + i] = R[i, j]
    
    return RI


def stacked(RI, rows, columns):
    '''
    Transforms an interwoven matrix into its stacked form.
    '''
    R = np.zeros((rows, columns))
    for i in range(0, rows):
        for j in range(0, columns):
            R[i, j] = RI[(j * rows) + i]

    return R


def resid(U, dU, PP, t, dt):
    '''
    Calculates the right hand side of the dU equation.
    '''
    r = np.zeros((PP.vars, N + 1))
    if PP.case == 1:
        WV = U[0]
        WP = U[1]
        dV = dU[0]
        dP = dU[1]

        for i in range(0, N + 1):
            dx = PP.x[i] - PP.x[i-1]
            r[0][i] = (1/dt) * dV[i] + (1/dx) * (WP[i] - WP[i-1])
        r[0][0] = WV[0]
        r[0][-1] = WV[-1]

        for j in range(0, N):
            r[1][j] = (1/dt) * dP[j] + (1/dx) * (WV[j+1] - WV[j])
        r[1][-1] = WP[-1]

        return r

    for j in range(0, PP.vars):
        PP.d = 1
        neg = 1
        if j == 1:
            PP.d = 0
            neg = -1/2
        for i in range(0, N):
            dx = PP.x[i + 1] - PP.x[i]
            WZmid = 0.5 * (U[0, i]+U[0,i+1])
            WQmid =  0.5 * (U[1,i]+U[1,i+1])

            r[j, i] += ((dx / dt) * ((dU[j][i] / 3) + (dU[j][i + 1] / 6))) + \
                ((PP.d / dx) * (U[j][i] - U[j][i+1])) - \
                    (neg * (f_x(WQmid, WZmid)) * (dx / 2))

            r[j, i + 1] += ((dx / dt) * ((dU[j][i] / 6) + (dU[j][i + 1] / 3))) + \
                ((PP.d / dx) * (U[j][i + 1] - U[j][i])) - \
                    (neg * (f_x(WQmid, WZmid)) * (dx / 2))

    r[0][0] = U[0][0] - u_a(t)

    return r


def step(PP, U, t, dt):
    '''
    Goes forward a single step to find the right-hand vector needed for
    the next value of dU.
    '''
    r = resid(U, np.zeros((PP.vars, N + 1)), PP, t, dt)
    
    c = create_C(U, PP, t, dt)
    r = interwoven(r, r.shape[0], r.shape[1])

    return solve_banded((PP.hbw, PP.hbw), c, -r)


def advance(PP, SimT, t_lst):
    '''
    Advances to a certain time step to find the corresponding approximation for U.
    '''
    if SimT.endTime == 0:
        return PP.u

    if t_lst[0] == 0:
        plotter(PP.x, [PP.u[0], PP.u[1]], \
                    ["v(x, t)", "p(x, t)"], \
                        f"Value of v and p at time t={t_lst[0]}.", \
                            "function at time t")
        t_lst.pop(0)

    U = copy.deepcopy(PP.u)
    while SimT.time < SimT.endTime:
        # creates our current step
        step1 = step(PP, U, SimT.time, SimT.dt)
        step1 = stacked(step1, 2, int(step1.shape[0]/2))
        S = U + step1

        # creates a double step to test against S
        step2 = step(PP, U, SimT.time, SimT.dt / 2)
        step2 = stacked(step2, 2, int(step2.shape[0]/2))
        D_first = U + step2
        step3 = step(PP, D_first, SimT.time, SimT.dt / 2)
        step3 = stacked(step3, 2, int(step3.shape[0]/2))
        D = D_first + step3

        # find the norm max of S and D
        e_i = np.amax(np.abs(S-D))

        if SimT.checker(e_i):
            U = (2 * D) - S

        if SimT.time >= t_lst[0]:
            if PP.case == 0:
                plotter(PP.x, [U[0], U[1]], \
                    ["z(x, t)", "q(x, t)"], \
                        f"Value of z and q at time t={t_lst[0]}.", \
                            "function at time t")
            else:
                plotter(PP.x, [U[0], U[1]], \
                    ["v(x, t)", "p(x, t)"], \
                        f"Value of v and p at time t={t_lst[0]}.", \
                            "function at time t")
            t_lst.pop(0)

    return U


def create_C(U, PP, t, dt):
    '''
    Creates the Jacobian matrix needed for banded calculations.
    '''
    Rbase = resid(U, np.zeros((PP.vars, N + 1)), PP, t, dt)
    bw = 1 + (2 * PP.hbw)
    c = np.zeros((bw, 2 * (N + 1)))
    for i in range(0, bw):
        dU = np.zeros(2*(N + 1))
        for j in range(i, 2*(N + 1), bw):
            dU[j] += epsilon
        dU = stacked(dU, 2, int(dU.shape[0]/2))
        R = (resid(U + dU, dU, PP, t, dt) - Rbase) / epsilon
        RI = interwoven(R, R.shape[0], R.shape[1])

        for k in range(i, 2*(N+1), bw):
            for l in range(0, bw):
                x = k - PP.hbw + l
                if x <= (2 * N) + 1:
                    c[l, k] = RI[x]

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
    # Example 1
    z = np.zeros(N + 1)
    q = np.ones(N + 1)

    case0 = Problem_Def(0, 1, z, q, 3, 2, 0)
    
    t_lst = [i for i in range(1, 11)]

    ST = simTime(0.0001, 0.01, 1.25, 0.8, 0.0001, 0.1, 10)
    U = advance(case0, ST, t_lst)

    plotter(np.array(ST.time_lst), [np.log10(np.array(ST.dt_lst))], \
        ["time vs. log10(dt)"], "Time vs. log10(dt) plot for Case 0.", "log10(dt)", "time")

 
    # Example 2
    v = np.zeros(N + 1)
    p = create_p()

    case1 = Problem_Def(0, 0, v, p, 1, 2, 1)

    t_lst = [0, 5, 13]

    ST = simTime(0.05, 0.01, 1.25, 0.8, 0.05, 0.05, 13)
    U = advance(case1, ST, t_lst)