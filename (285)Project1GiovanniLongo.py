import numpy as np
import matplotlib.pyplot as plt

def W_plot(W_lst, name_lst, dx, title):
	'''
	Takes in a list of the desired vectors we want to plot and returns
	a plot with all plotted functions. Also takes in a list of names
	for the respective vectors to plot.
	'''
	# create a list of indexes to match against the W vectors
	x_lst = np.arange(0, 1 + dx, dx)
	for i in range(0, len(W_lst)):
		plt.plot(x_lst, W_lst[i], label=name_lst[i])
		
	plt.xlabel("x-values")
	plt.ylabel("U(x,t)")
	plt.title(title)
	plt.legend(loc = 'best')
	plt.show()
	plt.clf()
	
	return x_lst


def find_W(N, dt, dx, v, U, t, case=0):
	'''
	Takes in parameters to return the desired approximation of u(x,t) at 
	time t.
	'''
	# return our initial vector at t=0
	if t == 0:
		return U
		
	step = dt
	# If we have set the case number to 0, we run a typical upwind, 
	# else we run it with the included BFECC. Run at each step dt until
	# we have reached our desired value of t.
	while step <= t:
		if case == 0:
			W = upwind(N, dt, dx, v, U)
		else:
			W = BFECC(N, dt, dx, v, U)
		step += dt
		U = W
		
	return W
			

def upwind(N, dt, dx, v, U):
	'''
	Takes in parameters to approximate the next step of u(x, t) via the 
	upwinding process.
	'''
	v1 = v
	W = []
	x = 0
	for i in range(0, N):
		# if we have hardcoded our v to be a string, we know that we are on
		# case 5
		if v1 == "v":
			x += dx
			v = find_v(x, dx)
		# find the CFL
		frac = v * (dt/dx)
		if frac < 0:
			k = i + 1
			frac = -frac
		else:
			k = np.mod(i - 1, N)
		c = frac * (U[k] - U[i])
		W.append(U[i] + c)
	# set the last value equal to the initial value since u is periodic 
	# in x
	W.append(W[0])
		
	return W
	
	
def make_U0(N):
	'''
	Constructs the initial vector U_0 based on our function g(x).
	'''
	U = []
	k = 0
	# create an initial vector of length N+1 where g(x) is given
	for i in range(0, N + 1):
		if k <= (1/20):
		 U.append(20 * k)
		elif k <= (1/10):
		 U.append(2 - (20 * k))
		else:
		 U.append(0)
		k += (1/N)
		 
	return U
	

def BFECC(N, dt, dx, v, U):
	'''
	Uses the back and forth error compensation and correction approach
	to find the desired approximation for u(x,t).
	'''
	G = upwind(N, dt, dx, v, U)
	B = upwind(N, -dt, dx, v, G)
	D = (1/2) * np.subtract(np.array(U), np.array(B))
	C = np.add(np.array(U), D)
	
	return upwind(N, dt, dx, v, list(C)) 


def find_v(x, dx):
	'''
	A function for case 5 to find the specific value of v(x) needed at 
	a given x value. A value dx is added for float issues.
	'''
	# the function v(x) as it is defined in case 5
	if x <= (1/4):
		return 1
	elif x <= (1/2):
		return 1 - (2*(x-(1/4)))
	elif x <= (3/4):
		return (1/2)
	else:
		return (1/2) + (2*(x-(3/4)))


if __name__ == "__main__":
	# CASE 0
	U = make_U0(80)
	t_lst = [0, 1/4, 1/2, 1]
	W_lst = []
	for t in t_lst:
		W = find_W(80, 1/80, 1/80, 1, U, t)
		W_lst.append(W)
		
	name_lst = ["t=0", "t=1/4", "t=1/2", "t=1"]
	l = W_plot(W_lst, name_lst, 1/80, \
		"Plot of the Approximation U(x,t) with dx=dt=1/80 and v(x)=1.")
	
	
	# CASE 1
	U = make_U0(80)
	t_lst = [0, 1/4, 1/2, 1]
	W_lst = []
	for t in t_lst:
		W = find_W(80, 1/80, 1/80, -1, U, t)
		W_lst.append(W)
		
	name_lst = ["t=0", "t=1/4", "t=1/2", "t=1"]
	l = W_plot(W_lst, name_lst, 1/80, \
		"Plot of the Approximation U(x,t) with dx=dt=1/80 and v(x)=-1.")
	
	
	# CASE 2
	U = make_U0(80)
	t_lst = [0, 1/4, 1/2, 1]
	W_lst = []
	for t in t_lst:
		W = find_W(80, 1/160, 1/80, 1, U, t)
		W_lst.append(W)
		
	name_lst = ["t=0", "t=1/4", "t=1/2", "t=1"]
	l = W_plot(W_lst, name_lst, 1/80, \
		"Plot of the Approximation U(x,t) with dx=1/80, dt=1/160, and v(x)=1.")
		
		
	# CASE 3
	U = make_U0(80)
	t_lst = [0, 1/4, 1/2, 1]
	W_lst = []
	for t in t_lst:
		W = find_W(80, 1/160, 1/80, 1, U, t, 1)
		W_lst.append(W)
		
	name_lst = ["t=0", "t=1/4", "t=1/2", "t=1"]
	l = W_plot(W_lst, name_lst, 1/80, \
		"Plot of the Approximation U(x,t) with dx=1/80, dt=1/160, and v(x)=1 using BFECC.")	
		
	
	# CASE 4
	U = make_U0(320)
	t_lst = [0, 1/4, 1/2, 1]
	W_lst = []
	for t in t_lst:
		W = find_W(320, 1/640, 1/320, 1, U, t, 1)
		W_lst.append(W)
		
	name_lst = ["t=0", "t=1/4", "t=1/2", "t=1"]
	l = W_plot(W_lst, name_lst, 1/320, \
		"Plot of the Approximation U(x,t) with dx=1/320, dt=1/640, and v(x)=1 using BFECC.")
		
	
	# CASE 5
	U = make_U0(320)
	t_lst = []
	t = 0
	while t < (3/4) + np.log(2):
		t_lst.append(t)
		t += 0.3
	
	W_lst = []
	for t in t_lst:
		W = find_W(320, 1/640, 1/320, "v", U, t, 1)
		W_lst.append(W)
	
	name_lst = ["t=0", "t=0.3", "t=0.6", "t=0.9", "t=1.2"]
	W_plot(W_lst, name_lst, 1/320, \
		"Plot of the Approximation U(x,t) with dx=1/320, dt=1/640, and v(x) using BFECC.")
	
