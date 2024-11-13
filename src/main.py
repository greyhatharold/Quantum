import numpy as np
from core.quantum_solver import QuantumSolver
from visualization.stationary_plotter import StationaryStatePlotter
from visualization.evolution_animator import TimeEvolutionAnimator
from utils.initial_states import InitialStateFactory
from gui.quantum_gui import QuantumGUI
import tkinter as tk

def run_quantum_simulation(potential_type="harmonic"):
    """
    Run quantum simulation with specified potential
    
    Parameters:
    -----------
    potential_type : str
        Type of potential to simulate ("harmonic" or "well")
    """
    # Initialize solver
    solver = QuantumSolver(x_min=-10, x_max=10, n_points=1000)
    
    # Set up potential
    if potential_type == "harmonic":
        omega = 1.0  # Angular frequency
        V = lambda x: 0.5 * solver.m * omega**2 * x**2
    elif potential_type == "well":
        V = lambda x: np.where(np.abs(x) < 5, 0, 100)
    else:
        raise ValueError("Unsupported potential type")
    
    solver.set_potential(V)
    
    # Solve for stationary states
    print("Solving for stationary states...")
    energies, eigenstates = solver.solve_stationary_states(n_states=5)
    
    # Plot stationary states
    print("Plotting stationary states...")
    StationaryStatePlotter.plot_states(solver)
    
    # Create initial state
    print("Creating initial state...")
    initial_state = InitialStateFactory.gaussian_wave_packet(
        solver,
        x0=-2.0,  # Initial position
        p0=1.0,   # Initial momentum
        sigma=0.5  # Width
    )
    
    # Time evolution
    print("Computing time evolution...")
    t = np.linspace(0, 10, 200)
    evolved_states = solver.time_evolution(initial_state, t)
    
    # Animate time evolution
    print("Animating time evolution...")
    TimeEvolutionAnimator.animate(solver, evolved_states, t, interval=50)


if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumGUI(root)
    root.mainloop()