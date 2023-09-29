import os
import re
import json
from copy import deepcopy
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from .strogatz_equations import equations


config = {
    "t_span": (0, 10),  # time span for integration
    "method": "LSODA",  # method for integration
    "rtol": 1e-5,  # relative tolerance (let's be strict)
    "atol": 1e-7,  # absolute tolerance (let's be strict)
    "first_step": 1e-6,  # initial step size (let's be strict)
    "t_eval": np.linspace(0, 10, 150),  # output times for the solution
    "min_step": 1e-10,  # minimum step size (only for LSODA)
}

matplotlib_rc = {
#'text': {'usetex': True},
'font': {'size': '16', 'family': 'serif'},#, 'serif': 'Palatino'},
'figure': {'titlesize': '20'},
'axes': {'titlesize': '22', 'labelsize': '28'},
'xtick': {'labelsize': '22'},
'ytick': {'labelsize': '22'},
'lines': {'linewidth': 3, 'markersize': 10},
'grid': {'color': 'grey', 'linestyle': 'solid', 'linewidth': 0.5},
}

def wrap_text(description, line_length):
    words = description.split()
    lines = []
    current_line = []

    for word in words:
        # If adding the next word doesn't exceed the line length or the current line is empty
        if len(' '.join(current_line) + ' ' + word) <= line_length or not current_line:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]

    # Add the last line if there's any
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)



def validate_equations(equations):
    """Validates the equations to make sure they are in the correct format.
    
    These are just a bunch of basic checks, which would probably all throw errors
    when trying to solve them anyway, but were useful to get the equations right
    in the beginning.
    """
    for eq_dict in equations:
        eq_string = eq_dict['eq']
        dim = eq_dict['dim']
        consts_values = eq_dict['consts']
        init_values = eq_dict['init']
        id = eq_dict['id']
        individual_eqs = eq_string.split('|')
        if len(individual_eqs) != dim:
            print(f"Error in equation {id}: The number of equations does not match the dimension.")

        highest_x_index = max([int(x[2:]) for x in re.findall(r'x_\d+', eq_string)])
        if highest_x_index + 1 != dim:
            pass #print(f"Warning in equation {id}: Found x_{highest_x_index} as highest index, but the dimension is {dim}.")
        
        const_indices = [int(c[2:]) for c in re.findall(r'c_\d+', eq_string)]
        if len(const_indices) > 0:
            highest_const_index = max(const_indices)
            for j in range(highest_const_index + 1):
                if f'c_{j}' not in eq_string:
                    print(f"Warning in equation {id}: c_{j} not appearing even though c_{highest_const_index} does.")
        for j, consts in enumerate(consts_values):
            if len(set(const_indices)) != len(consts):
                print(f"Warning in equation {id}, constants {j}: The number of constants does not match the number of constants in the equations.")

        for j, init in enumerate(init_values):
            if len(init) != dim:
                print(f"Error in equation {id}, init {j}: The number of initial values does not match the dimension of the equation.")
    print("VALIDATION DONE")


def process_equations(equations):
    """Create sympy expressions for each of the equations (and their different parameter values).
    We directly add the list of expressions to each dictionary.
    """
    validate_equations(equations)
    for eq_dict in equations:
        substituted_fns = create_substituted_functions(eq_dict)
        eq_dict['substituted'] = substituted_fns
    print("PROCESSING DONE")


def create_substituted_functions(eq_dict):
    """For a given equation, create sympy expressions where the different parameter values have been substituted in."""
    eq_string = eq_dict['eq']
    consts_values = eq_dict['consts']
    individual_eqs = eq_string.split('|')
    const_symbols = sp.symbols([f'c_{i}' for i in range(len(consts_values[0]))])
    parsed_eqs = [sp.sympify(eq) for eq in individual_eqs]

    substituted_fns = []
    for consts in consts_values:
        const_subs = dict(zip(const_symbols, consts))
        substituted_fns.append([eq.subs(const_subs) for eq in parsed_eqs])
    return substituted_fns


def save_to_disk(equations, filename):
    """Save the equations (including substituted sympy expressions) to disk"""
    store = deepcopy(equations)
    # Can't serialize sympy expressions, so convert them to strings
    for eq_dict in store:
        for i, fns in enumerate(eq_dict['substituted']):
            for j, fn in enumerate(fns):
                eq_dict['substituted'][i][j] = str(fn)
    with open(filename, 'w') as f:
        json.dump(store, f)
    print("SAVING DONE")


def solve_equations(equations, config):
    """Solve all equations for a given config.
    
    We add the solutions to each of the equations dictionary as a list of list of solution dictionaries.
    The list of list represents (number of parameter settings x number of initial conditions).
    """
    for eq_dict in tqdm(equations):
        eq_dict['solutions'] = []
        var_symbols = sp.symbols([f'x_{i}' for i in range(eq_dict['dim'])])
        for i, fns in enumerate(eq_dict['substituted']):
            eq_dict['solutions'].append([])
            callable_fn = lambda t, x: np.array([f(*x) for f in [sp.lambdify(var_symbols, eq, 'numpy') for eq in fns]])
            for initial_conditions in eq_dict['init']:
                sol = solve_ivp(callable_fn, **config, y0=initial_conditions)
                sol_dict = {
                    "success": sol.success,
                    "message": sol.message,
                    "t": sol.t.tolist(),
                    "y": sol.y.tolist(),
                    "nfev": int(sol.nfev),
                    "njev": int(sol.njev),
                    "nlu": int(sol.nlu),
                    "status": int(sol.status),
                }
                if sol.status != 0:
                    print(f"Error in equation {eq_dict['id']}: {eq_dict['eq_description']}, constants {i}, initial conditions {initial_conditions}: {sol.message}")
                sol_dict['consts'] = eq_dict['consts'][i]
                sol_dict['init'] = initial_conditions
                eq_dict['solutions'][i].append(sol_dict)
    print("SOLVING DONE")


def plot_trajectories(equations):
    """Create plots for all solved trajectories for all equations for visual inspection."""
    # Create a directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Loop over all equations
    for eq_dict in tqdm(equations):
        eq_id = eq_dict['id']
        description = eq_dict['eq_description']
        title_words = description.split()[:4]
        filename_words = description.split()[:6]

        # Title for the plots
        plot_title = ' '.join(title_words)
        # Filename for the .pdf file
        plot_filename = '_'.join(filename_words).lower()
        plot_filename = plot_filename.replace(' ', '_').replace(r'/', '_') + '.pdf'
        
        # Get the number of unique combinations of constants and initial conditions
        n_consts = len(eq_dict['consts'])
        n_inits = len(eq_dict['init'])
        
        # Create subplots
        fig, axs = plt.subplots(n_inits, n_consts, figsize=(n_consts*5, n_inits*5), constrained_layout=True);
        axs = np.atleast_2d(axs)
        if n_consts == 1:
            axs = axs.T
        
        # Iterate over each combination of constants and initial conditions
        for i in range(n_inits):
            for j in range(n_consts):
                # Extract the corresponding solution
                solution = eq_dict['solutions'][j][i]
                times = solution['t']
                y_values = solution['y']

                # Plot each dimension in the same subplot
                for dim, y in enumerate(y_values):
                    axs[i, j].plot(times, y, label=f"x_{dim}");

                axs[i, j].set_title(plot_title + f" (init {i+1}, const {j+1})")
                axs[i, j].set_xlabel('t')
                axs[i, j].legend()
        
        # Save the plot as .pdf
        fig.savefig(f'plots/{eq_id}_{plot_filename}', format='pdf');
        plt.close(fig)
    print("PLOTTING DONE")


def plot_all_equations(equations, line_length):
    """Create a single plot containing all solved trajectories for all equations."""

    # Restore defaults first
    matplotlib.rcdefaults()

    # Apply changes
    for k, v in matplotlib_rc.items():
        matplotlib.rc(k, **v)

    # Create subplots 11x5
    rows = 9
    cols = 7
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True, sharex=True);

    # Loop over all equations
    for k, eq_dict in enumerate(tqdm(equations)):
        eq_id = eq_dict['id']
        description = eq_dict['eq_description']
        plot_title = wrap_text(description, line_length)

        # Filename for the .pdf file
        plot_filename = 'odebench_trajectories.pdf'
        
        # Iterate over each combination of constants and initial conditions
        solution = eq_dict['solutions'][0][0]
        times = solution['t']
        y_values = solution['y']

        # Convert single index k to i, j indices with i indicating rows and j columns
        i, j = k // cols, k % cols

        # Plot each dimension in the same subplot
        for dim, y in enumerate(y_values):
            axs[i, j].plot(times, y, label=f"x_{dim}");
        axs[i, j].set_title(plot_title)

    for j in range(cols):
        axs[-1, j].set_xlabel('t')
    
    # Save the plot as .pdf
    fig.savefig(plot_filename, format='pdf');
    plt.close(fig)
    print("PLOTTING ALL EQUATIONS DONE")

def plot_prediction(dstr, equation, noise=0, subsampling=0, seed=0, save=False):

    for k, v in matplotlib_rc.items():
        matplotlib.rc(k, **v)

    name = equation['eq_description']
    id = equation['id']
    solution = equation['solutions'][0][0]
    time = np.array(solution['t'])
    trajectory = np.array(solution['y']).T
    dimension = trajectory.shape[1]
    
    # solve ODE
    if seed is not None:
        np.random.seed(seed)

    plot_time = np.linspace(min(time),max(time),1000)     # time
    original_time = time.copy()
    original_trajectory = trajectory.copy()
    time, trajectory = original_time.copy(), original_trajectory.copy()
    indices_to_drop = np.random.choice(len(time), int(subsampling*len(time)), replace=False)
    time = np.delete(time, indices_to_drop, axis=0)
    trajectory = np.delete(trajectory, indices_to_drop, axis=0)
    trajectory *= (1+noise*np.random.randn(*trajectory.shape))

    plt.figure(figsize=(5,5))
    for dim in range(dimension):
        plt.plot(original_time, original_trajectory[:,dim], color=f'C{dim}', alpha=.2, lw=10)
        plt.plot(time, trajectory[:,dim], ls="None", marker='o', alpha=.3, color=f'C{dim}')
    plt.grid(False)

    candidates = dstr.fit(time, trajectory, verbose=False)

    for i, tree in enumerate(candidates[0][:1]):

        pred_trajectory = dstr.predict(plot_time, trajectory[0])
        try:
            for dim in range(dimension):
                plt.plot(plot_time, pred_trajectory[:,dim])
        except: 
            import traceback
            print(traceback.format_exc())

    ax = plt.gca()
    ax.text(0.5, 1.2, wrap_text(name, 28), transform=ax.transAxes, fontsize=18, 
            verticalalignment='center', horizontalalignment='center')
    #plt.title()
    plt.yticks([])
    plt.xlabel('t')
    plt.tight_layout()
    if save: plt.savefig(f'figs/odebench/odebench_{id}.pdf')
    plt.show()

    return trajectory, pred_trajectory


if __name__ == '__main__':
    process_equations(equations)
    solve_equations(equations, config)
    save_to_disk(equations, 'solutions.json')
    plot_trajectories(equations)
    plot_all_equations(equations, line_length=32)
