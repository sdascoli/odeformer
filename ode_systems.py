import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import traceback

def lotka_volterra(X, t, a=3, b=3, c=3, d=3):
    return np.array([ a*X[0] - b*X[0]*X[1] ,
                     -c*X[1] + d*X[0]*X[1] ])
lv_system = {
    'name':'lotka volterra', 
    'function':lotka_volterra, 
    'labels':['Rabbits', 'Foxes'],
    'n_points':100,
    'tmax':10,
    'init':[.5, .5]
}

def lorenz(X, t, s=10, r=28, b=2.667):
    x, y, z = X
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot
lorenz_system = {
    'name':'lorenz attractor', 
    'function':lorenz, 
    'labels':['x','y','z'],
    'n_points':100,
    'tmax':4,
    'init':[10,20,20]
}

def apoptosis(X, t, k0=.1, k1=.6, km1=.2, km3=7.95, kd=.05, k2=0.4, jm1=.1, jm3=2, j1=0.1, j2=0.1):
    x, y, z = X
    v0     = k0
    v1     = k1 * z * (j1 + y)
    vm1    = km1 * y / (jm1 + y)
    v2     = k2 * y * x / (j2 + x)
    vm3    = km3 * x * y / (jm3 + y)
    dx     = v0 - v2 - kd * x
    dy     = v1 - vm1 - vm3
    dz     = -dy
    return dx, dy, dz
apoptosis_system = {
    'name':'apoptosis', 
    'function':apoptosis, 
    'labels':['x','y','z'],
    'n_points':100,
    'tmax':10,
    #'init':[0.248, 0.0973, 0.0027]
    'init':[0.1, 0.0973, 0.0027]
}

def robertson(X, t):
    """ODEs for Robertson's chemical reaction system."""
    x, y, z = X
    xdot = -0.04 * x + 1.e4 * y * z
    ydot = 0.04 * x - 1.e4 * y * z - 3.e7 * y**2
    zdot = 3.e7 * y**2
    return xdot, ydot, zdot
robertson_system = {
    'name':'robertson', 
    'function':robertson, 
    'labels':['x','y','z'],
    'n_points':100,
    'tmax':500,
    'init':[1, 0.1, 0.1] 
}

def hyperbolic(X, t):
    x, y = X
    xdot = -0.05* x
    ydot = x**2-y**2
    return xdot, ydot
hyperbolic_system = {
    'name':'hyperbolic system', 
    'function':hyperbolic, 
    'labels':['x','y'],
    'n_points':100,
    'tmax':10,
    'init':[-1, -0.5] 
}

def cubic(X, t):
    x, y = X
    xdot = -0.1 * x**3 + 2*y**3 
    ydot = -2 * x**3 - 0.1*y**3
    return xdot, ydot
cubic_system = {
    'name':'cubic', 
    'function':cubic, 
    'labels':['x','y'],
    'n_points':100,
    'tmax':10,
    'init':[-1, -0.5] 
}

def vdp(X, t):
    x, y = X
    xdot = y
    ydot = -x +2*y - 2*x**2*y
    return xdot, ydot
vdp_system = {
    'name':'van der pol oscillator', 
    'function':vdp, 
    'labels':['x','y'],
    'n_points':100,
    'tmax':10,
    'init':[.2, 1] 
}

def dno(X, t):
    x, y, z = X
    xdot = y
    ydot = -3*np.sin(x)-0.04*y
    zdot = 0.04*y**2
    return xdot, ydot, zdot
dno_system = {
    'name':'damped nonlinear oscillator', 
    'function':dno, 
    'labels':['q','p', 'S'],
    'n_points':100,
    'tmax':10,
    'init':[.8, -.7, .1] 
}

def plot_prediction(dstr, system=lv_system, noise=0, subsampling=0, seed=0):

    name, function, labels, y0, n_points, tmax = system['name'], system['function'], system['labels'], system['init'], system['n_points'], system['tmax']
    dimension = len(y0)
    
    # solve ODE
    np.random.seed(seed)
    time = np.linspace(1,tmax,n_points)     # time
    trajectory = integrate.odeint(function, y0, time)
    plot_time = np.linspace(min(time),max(time),1000)     # time
    original_time = time.copy()
    original_trajectory = trajectory.copy()
    time, trajectory = original_time.copy(), original_trajectory.copy()
    rng = np.random.RandomState(seed)
    indices_to_drop = rng.choice(len(time), int(subsampling*len(time)), replace=False)
    time = np.delete(time, indices_to_drop, axis=0)
    trajectory = np.delete(trajectory, indices_to_drop, axis=0)
    trajectory *= (1+noise*np.random.randn(*trajectory.shape))

    plt.figure()
    for dim in range(dimension):
        plt.plot(original_time, original_trajectory[:,dim], color=f'C{dim}', alpha=.2, lw=10)
        plt.plot(time, trajectory[:,dim], ls="None", marker='o', alpha=.5, color=f'C{dim}')
    if system == lv_system:
        plt.plot([],[], color='k', alpha=.2, lw=10, label='Ground truth')
        plt.plot([],[], color='k', alpha=.5, ls="None", marker='o', label='Noisy data')
        plt.plot([],[], color='k', label='Predicted')
        plt.legend()

    candidates = dstr.fit(time, trajectory, verbose=False)

    for i, tree in enumerate(candidates[0][:1]):

        pred_trajectory = dstr.predict(plot_time, trajectory[0])
        try:
            for dim in range(dimension):
                plt.plot(plot_time, pred_trajectory[:,dim])# label=labels[dim])
        except: 
            print(traceback.format_exc())
    plt.xlabel('Time')
    plt.tight_layout()
    #plt.savefig(savedir+f'{name}.pdf')
    plt.title(name)
    plt.show()

    return trajectory, pred_trajectory


my_ode_systems = [lv_system, hyperbolic_system, cubic_system, vdp_system, dno_system, lorenz_system] #apoptosis_system, robertson_system
