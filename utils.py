import numpy as np
import pickle
import os
import math
from numpy import linalg
from scipy.integrate import cumtrapz
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
tf = 15
V_max = 0.5
om_max = 1

# time
dt = 0.005
N = int(tf/dt)
t = dt*np.array(range(N+1)) 

# Initial conditions
x_0 = 0
y_0 = 0
V_0 = V_max
th_0 = -np.pi/2
xd_0 = V_0*np.cos(th_0)
yd_0 = V_0*np.sin(th_0)

# Final conditions
x_f = 5
y_f = 5
V_f = V_max
th_f = -np.pi/2
xd_f = V_f*np.cos(th_f)
yd_f = V_f*np.sin(th_f)


def car_dyn(x, t, ctrl, noise):
    u_0 = ctrl[0] + noise[0]
    u_1 = ctrl[1] + noise[1]
    dxdt = [u_0 * np.cos(x[2]),
            u_0 * np.sin(x[2]),
            u_1]
    return dxdt


def wrapToPi(a):
    if isinstance(a, list):  # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2 * np.pi) - np.pi for x in a]
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_flip(z0):
    flip = 0
    if z0[-1] < 0:
        tf = -z0[-1]
        flip = 1
    else:
        tf = z0[-1]
    return flip, tf


from six.moves import cPickle as pickle  # for performance


def get_folder_name(filename):
    return '/'.join(filename.split('/')[:-1])


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def save_dict(di_, filename_):
    maybe_makedirs(get_folder_name(filename_))
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def differential_flatness_trajectory():
    '''
    This function solves a system of equations and computes the state trajectory.
    and control history (V (t), om(t)). 
    Outputs:
        traj: a numpy array of size [T, state_dim] where T is the number of time steps, and state_dim is 6. 
        The state ordering needs to be [x,y,th,xd,yd,xdd,ydd]
    
    HINT: You may find the function linalg.solve useful
    '''
    ########## Code starts here ##########
    # Linear equations
    A = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  #x_0
                 [1, tf, pow(tf,2), pow(tf,3), 0, 0, 0, 0], #x_f
                 [0, 1, 0, 0, 0, 0, 0, 0], #xd_0
                 [0, 1, 2*tf, 3*pow(tf,2), 0, 0, 0, 0], #xd_f
                 [0, 0, 0, 0, 1, 0, 0, 0], #y_0
                 [0, 0, 0, 0, 1, tf, pow(tf,2), pow(tf,3)], #y_f
                 [0, 0, 0, 0, 0, 1, 0, 0], #yd
                 [0, 0, 0, 0, 0, 1, 2*tf, 3*pow(tf,2)]#yd_f
                 ])
    B = np.array([x_0, x_f, xd_0, xd_f, y_0, y_f, yd_0, yd_f])
    coeff = linalg.solve(A,B)
    traj = np.zeros((N+1,7))
    V = np.zeros((N+1))
    om = np.zeros((N+1))
    # Compute trajectory, store in traj, format: [x,y,th,xd,yd,xdd,ydd]
    for i in range(N+1): #0....N
        traj[i,0] = coeff[0] + coeff[1]*t[i] + coeff[2]*pow(t[i],2) + coeff[3]*pow(t[i],3)
        traj[i,1] = coeff[4] + coeff[5]*t[i] + coeff[6]*pow(t[i],2) + coeff[7]*pow(t[i],3)
        traj[i,3] = coeff[1] + coeff[2]*2*t[i] + coeff[3]*3*pow(t[i],2)
        traj[i,4] = coeff[5] + coeff[6]*2*t[i] + coeff[7]*3*pow(t[i],2)
        traj[i,5] = coeff[2]*2 + coeff[3]*6*t[i]
        traj[i,6] = coeff[6]*2 + coeff[7]*6*t[i]
        traj[i,2] = np.arctan2(traj[i,4], traj[i,3])
        V[i] = linalg.norm(traj[i,3:5])
        om[i] = (traj[i,6]*traj[i,3] - traj[i,5]*traj[i,4])/(V[i]**2)
    ########## Code ends here ##########
    return traj, V, om

def compute_arc_length(V, t):
    '''
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time t[i]. This has length T.

    HINT: Use the function cumtrapz
    HINT: This should take one line
    '''
    ########## Code starts here ##########
    s = cumtrapz(V, t, initial=0)
    ########## Code ends here ##########
    return s

def rescale_V(V, om):
    '''
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.
    HINT: at each timestep V_tilde should be computed as a minimum of the original value V, and values required to ensure _both_ constraints are satisfied

    HINT: This should take one line
    '''
    ########## Code starts here ##########
    V_tilde = np.minimum(np.minimum(abs(V), V_max), om_max*abs(V/om))*np.sign(V)
    ########## Code ends here ##########
    return V_tilde


def compute_tau(V_tilde, s):
    '''
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a vector of scaled velocities of length T.
        s: a vector of arc-length of length T.
    Output:
        tau: the new time history as a function of time. tau[i] is the time at s[i]. This has length T.

    HINT: Use the function cumtrapz
    HINT: This should take one line
    '''
    ########## Code starts here ##########
    tau = cumtrapz(1/V_tilde, s, initial=0)
    ########## Code ends here ##########
    return tau

def rescale_om(V, om, V_tilde):
    '''
    This function computes the rescaled om control
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    HINT: This should take one line.
    '''
    ########## Code starts here ##########
    om_tilde = om*V_tilde/V
    ########## Code ends here ##########
    return om_tilde
