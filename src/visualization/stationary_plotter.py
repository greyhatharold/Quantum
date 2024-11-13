import numpy as np
import matplotlib.pyplot as plt

class StationaryStatePlotter:
    @staticmethod
    def plot_states(solver, mode='2d', fig=None):
        if mode == '2d':
            return StationaryStatePlotter._plot_2d(solver, fig)
        else:
            return StationaryStatePlotter._plot_3d(solver, fig)

    @staticmethod
    def _plot_3d(solver, fig=None):
        """3D visualization using matplotlib"""
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
            
        n_states = solver.eigenstates.shape[1]
        
        for i in range(n_states):
            if i > 0:
                fig.clear()
            
            ax = fig.add_subplot(111, projection='3d')
            state = solver.eigenstates[:, i].reshape(len(solver.y), len(solver.x))
            probability = np.abs(state)**2
            
            surf = ax.plot_surface(solver.X, solver.Y, probability,
                                 cmap='viridis', antialiased=True)
            
            ax.set_title(f'Eigenstate {i}, E = {solver.energies[i]:.2f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Probability Density')
            fig.colorbar(surf)
            
            fig.tight_layout()
            
        return fig

    @staticmethod
    def _plot_2d(solver, fig=None):
        """2D matplotlib plotting"""
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
            
        n_states = solver.eigenstates.shape[1]
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Plot wavefunctions
        for i in range(n_states):
            ax1.plot(solver.x, solver.eigenstates[:, i] + solver.energies[i],
                    label=f'E_{i} = {solver.energies[i]:.2f}')
        ax1.plot(solver.x, solver.V.diagonal(), 'k--', label='Potential')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Energy / Wavefunction')
        ax1.legend()
        ax1.grid(True)
        
        # Plot probability densities
        for i in range(n_states):
            ax2.plot(solver.x, np.abs(solver.eigenstates[:, i])**2,
                    label=f'n = {i}')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        return fig 
        plt.tight_layout()
        plt.show() 