import numpy as np
from src.core.quantum_solver import QuantumSolver
from src.visualization.stationary_plotter import StationaryStatePlotter
from src.visualization.evolution_animator import TimeEvolutionAnimator
from src.utils.initial_states import InitialStateFactory

def run_harmonic_oscillator_example():
    # Initialize solver
    solver = QuantumSolver(x_min=-10, x_max=10, n_points=1000)
    
    # Set up harmonic oscillator potential
    omega = 1.0
    V = lambda x: 0.5 * solver.m * omega**2 * x**2
    solver.set_potential(V)
    
    # Solve for stationary states
    solver.solve_stationary_states(n_states=5)
    
    # Plot stationary states
    StationaryStatePlotter.plot_states(solver)
    
    # Create initial state
    initial_state = InitialStateFactory.gaussian_wave_packet(
        solver, x0=-2.0, p0=1.0, sigma=0.5
    )
    
    # Time evolution
    t = np.linspace(0, 10, 200)
    evolved_states = solver.time_evolution(initial_state, t)
    
    # Animate time evolution
    TimeEvolutionAnimator.animate(solver, evolved_states, t, interval=50)

if __name__ == "__main__":
    run_harmonic_oscillator_example() 