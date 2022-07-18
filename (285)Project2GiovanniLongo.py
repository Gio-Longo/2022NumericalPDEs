from cmath import sqrt
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt


class BVP:
    '''
    A class to define the boundary value problem.
    '''
    def __init__(self, dx, I, d, e, x0, xn, dt, tfinal, case):
        self.sigma = create_mesh(dx, I)
        self.d = d
        self.e = e
        
        self.M = create_M(self.sigma, d, case)
        self.S = create_S(self.sigma, e, case)
        # create a combined matrix that will be used for the LHS
        self.C  = ((1/dt) * self.M) + ((dt/4) * self.S)
        
        # the boundaries have either a Dirichlet or Neumann condition
        self.x0 = x0
        self.xn = xn
        self.dt = dt
        self.tfinal = tfinal


    def step(self, U, V, t):
        '''
        Goes forward one step in the process to find the approximate
        solution U.
        '''
        W = (U + ((self.dt/2) * V))
        R = self.RHS(W)

        if self.x0 == "D":
            # if we have Dirichelet start point, then we ensure to have the
            # correct first value of our right-hand side based on flick(t)
            du_a = flick(t + self.dt) - U[0]
            dv_a = 2 * ((du_a/self.dt) - V[0])
            # edit the combined matrix to ensure dV[0] = dv_a
            self.C[1, 0] = 1
            self.C[0, 1] = 0
            R[0] = dv_a

        if self.xn == "D":
            # if we have Dirichelet end point, then we ensure to have the
            # correct final value of the right-hand side, which is 0
            self.C[1, -1] = 1
            self.C[2, -2] = 0
            R[-1] = 0

        dV = solve_banded((1,1), self.C, R)
        dU = self.dt * (V + ((1/2) * dV))
        
        return (dU, dV)
    
    
    def advance(self): 
        '''
        Goes from t=0 to tfinal to find the final approximate solution U.
        '''
        t = 0
        # for all cases, U and V have initial values as the 0 vector
        U = np.zeros(501)
        V = np.zeros(501)
        while t < (self.tfinal + self.dt):
            dU, dV = self.step(U, V, t)
            U += dU
            V += dV
            t += self.dt
            
        return U
    
    
    def RHS(self, W):
        '''
        Calculates the RHS column vector needed for the dV equation.
        '''
        S_neg = -self.S
        N = self.sigma.size
        R = np.zeros((N, 1))
        R[0] = (W[0] * S_neg[1,0]) + (W[1] * S_neg[2,0])
        R[-1] = (W[-2] * S_neg[0,-1]) + (W[-1] * S_neg[1,-1])
        for i in range(1, N - 1):
            R[i] = (W[i-1] * S_neg[0,i]) + (W[i] * S_neg[1,i]) + \
                (W[i+1] * S_neg[2,i])
        
        R = R.reshape(N)
        
        return R
        

def create_mesh(dx, I):
    '''
    Takes in the step dx and interval I to find the mesh
    for a given BVP class.
    '''
    a, b = I
    
    return np.arange(a, b + dx, dx)
    

def flick(t):
    '''
    Calculates the value of u(a,t) based on the given flick(t) equation.
    '''
    if t < (2 + 1/100):
        return (t * (2-t))**3
    
    return 0    
    

def create_M(sigma, d, case):
    '''
    Creates the tri-diagonal mass matrix as a 3 x (N+1) matrix.
    '''
    coef = d
    N = sigma.size
    M = np.zeros((3, N))
    for i in range(1, N):
        dx = sigma[i] - sigma[i-1]
        xmid = (sigma[i] + sigma[i-1]) / 2
        if d == "d":
            coef = d_x(xmid, case)
        M[1, i - 1] += coef * (dx/3)
        M[2, i - 1] = coef * (dx/6)
        M[0, i] = coef * (dx/6)
        M[1, i] = coef * (dx/6)
    
    return M
        
        
def create_S(sigma, e, case):
    '''
    Creates the tri-diagonal stiffness matrix as a 3 x (N+1) matrix.
    '''
    coef = e
    N = sigma.size
    S = np.zeros((3, N))
    for i in range(1, N):
        dx = sigma[i] - sigma[i-1]
        xmid = (sigma[i] + sigma[i-1]) / 2
        if e == "e":
            coef = e_x(xmid, case)
        S[1, i - 1] += coef / dx
        S[2, i - 1] = -1 * (coef / dx)
        S[0, i] = -1 * (coef / dx)
        S[1, i] = coef / dx
    
    return S
    
    
def d_x(x, case): 
    '''
    A function that modifies the density function for the mass matrix.
    '''
    if case == 2:
        return x
    if case == 3:
        return x ** 2
   
   
# for this project, d and e happen to have the same values for every x,
# but a function is created for each to allow for manipulation.
def e_x(x, case): 
    '''
    A function that modifies the elasticity function for the stiffness matrix.
    ''' 
    if case == 2:
        return x
    if case == 3:
        return x ** 2
    

def U_plot(sigma, U_lst, name_lst, title, ylabel):
	'''
	Takes in a list of the desired vectors we want to plot and returns
	a plot with all plotted functions. Also takes in a list of names
	for the respective vectors to plot.
	'''
	# create a list of indexes to match against the W vectors
	plt.plot(sigma, U_lst[0], label=name_lst[0])
	plt.plot(sigma, U_lst[1], label=name_lst[1])	

	plt.xlabel("x-values")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend(loc = 'best')
	plt.show()
	plt.clf()
	
	return None


if __name__ == "__main__":
    # Case 0
    case0_1 = BVP(1/50, (1, 11), 1, 1, "D", "D", 1/100, 5, 0)
    case0_2 = BVP(1/50, (1, 11), 1, 1, "D", "D", 1/100, 13, 0)
    
    U_plot(case0_1.sigma, [case0_1.advance(), case0_2.advance()], ["t=5", "t=13"], \
        "Approximation of u(x,t) with Dirichilet endpoints.", "U(x)")
    

    # Case 1
    case1_1 = BVP(1/50, (1, 11), 1, 1, "D", "N", 1/100, 5, 1)
    case1_2 = BVP(1/50, (1, 11), 1, 1, "D", "N", 1/100, 13, 1)
    
    U_plot(case1_1.sigma, [case1_1.advance(), case1_2.advance()], ["t=5", "t=13"], \
        "Approximation of u(x,t) with one Dirichilet and one Neumann endpoint.", \
            "U(x)")


    # Case 2
    case2_1 = BVP(1/50, (1, 11), "d", "e", "D", "D", 1/100, 5, 2)
    case2_2 = BVP(1/50, (1, 11), "d", "e", "D", "D", 1/100, 13, 2)

    sqrt_sig = np.sqrt(case2_1.sigma)

    U_21 = (sqrt_sig * case2_1.advance())
    U_22 = (sqrt_sig * case2_2.advance())

    U_plot(case2_1.sigma, [U_21, U_22], ["t=5", "t=13"], \
        "Plot of approximation U(x)sqrt(x) with Dirichilet endpoints and d(x)=e(x)=x.", \
            "U(x)sqrt(x)")


    # Case 3
    case3_1 = BVP(1/50, (1, 11), "d", "e", "D", "D", 1/100, 5, 3)
    case3_2 = BVP(1/50, (1, 11), "d", "e", "D", "D", 1/100, 13, 3)

    U_31 = (case3_1.sigma * case3_1.advance())
    U_32 = (case3_2.sigma * case3_2.advance())

    U_plot(case3_1.sigma, [U_31, U_32], ["t=5", "t=13"], \
        "Plot of approximation U(x)x with Dirichilet endpoints and d(x)=e(x)=x^2.", \
            "U(x)x")
    