import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TimeEvolutionAnimator:
    @staticmethod
    def animate(solver, evolved_states, t, interval=50, mode='2d', fig=None):
        if mode == '2d':
            return TimeEvolutionAnimator._animate_2d(solver, evolved_states, t, interval, fig)
        else:
            return TimeEvolutionAnimator._animate_3d(solver, evolved_states, t, interval, fig)

    @staticmethod
    def _animate_3d(solver, evolved_states, t, interval, fig=None):
        """3D animation using matplotlib"""
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            state_2d = evolved_states[frame].reshape(len(solver.y), len(solver.x))
            probability = np.abs(state_2d)**2
            
            surf = ax.plot_surface(solver.X, solver.Y, probability,
                                 cmap='viridis', antialiased=True)
            ax.set_title(f'Time: {t[frame]:.2f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Probability Density')
            return [surf]
        
        anim = FuncAnimation(fig, update, frames=len(t),
                           interval=interval, blit=False)
        
        return anim

    @staticmethod
    def _animate_2d(solver, evolved_states, t, interval, fig=None):
        """2D matplotlib animation"""
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        lines1 = []
        lines2 = []
        
        # Real and imaginary parts
        line_real, = ax1.plot([], [], 'b-', label='Real')
        line_imag, = ax1.plot([], [], 'r-', label='Imaginary')
        lines1.extend([line_real, line_imag])
        
        # Probability density
        line_prob, = ax2.plot([], [], 'g-', label='Probability')
        lines2.append(line_prob)
        
        # Set up axes
        ax1.set_xlim(solver.x[0], solver.x[-1])
        ax1.set_ylim(np.min(evolved_states.real)*1.5, np.max(evolved_states.real)*1.5)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Wavefunction')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlim(solver.x[0], solver.x[-1])
        ax2.set_ylim(0, np.max(np.abs(evolved_states)**2)*1.2)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        ax2.grid(True)
        
        def init():
            for line in lines1 + lines2:
                line.set_data([], [])
            return lines1 + lines2
        
        def animate(i):
            lines1[0].set_data(solver.x, evolved_states[i].real)
            lines1[1].set_data(solver.x, evolved_states[i].imag)
            lines2[0].set_data(solver.x, np.abs(evolved_states[i])**2)
            ax1.set_title(f'Time: {t[i]:.2f}')
            return lines1 + lines2
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                           interval=interval, blit=True)
        
        fig.tight_layout()
        return anim