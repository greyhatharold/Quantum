import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class QuantumSolver:
    def __init__(self, x_min=-10, x_max=10, y_min=None, y_max=None, 
                 n_points_x=100, n_points_y=None, ℏ=1.0, m=1.0):
        # Set up spatial grid (1D or 2D)
        self.x = np.linspace(x_min, x_max, n_points_x)
        self.dx = self.x[1] - self.x[0]
        self.dimensions = 1

        if y_min is not None and y_max is not None:
            self.dimensions = 2
            self.y = np.linspace(y_min, y_max, n_points_y or n_points_x)
            self.dy = self.y[1] - self.y[0]
            self.X, self.Y = np.meshgrid(self.x, self.y)

        self.ℏ = ℏ
        self.m = m
        
        # Create appropriate kinetic energy operator
        if self.dimensions == 1:
            self._setup_1d_kinetic()
        else:
            self._setup_2d_kinetic()

    def _setup_1d_kinetic(self):
        n = len(self.x)
        diag = np.ones(n) * (-2)
        off_diag = np.ones(n-1)
        self.K = sparse.diags([off_diag, diag, off_diag], [-1, 0, 1]) 
        self.K *= -self.ℏ**2 / (2 * self.m * self.dx**2)

    def _setup_2d_kinetic(self):
        nx, ny = len(self.x), len(self.y)
        N = nx * ny
        
        # Create 2D Laplacian using Kronecker products
        Dx = sparse.diags([-2, 1, 1], [0, 1, -1], shape=(nx, nx)) / self.dx**2
        Dy = sparse.diags([-2, 1, 1], [0, 1, -1], shape=(ny, ny)) / self.dy**2
        
        I_x = sparse.eye(nx)
        I_y = sparse.eye(ny)
        
        self.K = sparse.kron(I_y, Dx) + sparse.kron(Dy, I_x)
        self.K *= -self.ℏ**2 / (2 * self.m)

    def set_potential(self, V_func):
        if self.dimensions == 1:
            V_values = V_func(self.x)
        else:
            V_values = V_func(self.X, self.Y).flatten()
            
        self.V = sparse.diags([V_values], [0])
        self.H = self.K + self.V
    
    def solve_stationary_states(self, n_states=5):
        eigenvalues, eigenvectors = eigsh(self.H.tocsc(), k=n_states, which='SM')
        idx = np.argsort(eigenvalues)
        self.energies = eigenvalues[idx]
        self.eigenstates = eigenvectors[:, idx]
        return self.energies, self.eigenstates
    
    def time_evolution(self, initial_state, t):
        coefficients = np.zeros(len(self.energies), dtype=complex)
        for i in range(len(self.energies)):
            coefficients[i] = np.sum(np.conj(self.eigenstates[:, i]) * initial_state) * self.dx
        
        evolved_states = np.zeros((len(t), len(self.x)), dtype=complex)
        for i, time in enumerate(t):
            state = np.zeros_like(initial_state, dtype=complex)
            for j in range(len(self.energies)):
                state += coefficients[j] * self.eigenstates[:, j] * np.exp(-1j * self.energies[j] * time / self.ℏ)
            evolved_states[i] = state
        
        return evolved_states 
    
    def plot_animation_frame(self, fig, state):
        """Plot a single frame of the animation (for GUI)"""
        fig.clear()
        
        if self.dimensions == 1:
            self._plot_1d_frame(fig, state)
        else:
            self._plot_2d_frame(fig, state)
            
    def _plot_1d_frame(self, fig, state):
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Plot real and imaginary parts
        ax1.plot(self.x, state.real, 'b-', label='Real')
        ax1.plot(self.x, state.imag, 'r-', label='Imaginary')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Wavefunction')
        ax1.legend()
        ax1.grid(True)
        
        # Plot probability density
        ax2.plot(self.x, np.abs(state)**2, 'g-', label='Probability')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        
    def _plot_2d_frame(self, fig, state):
        ax = fig.add_subplot(111, projection='3d')
        
        # Reshape state for 2D plot
        state_2d = state.reshape(len(self.y), len(self.x))
        probability = np.abs(state_2d)**2
        
        # Create surface plot
        surf = ax.plot_surface(self.X, self.Y, probability, 
                             cmap='viridis', antialiased=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        fig.colorbar(surf)
        
        fig.tight_layout()

    def gaussian_well(self, x0, y0, depth, width):
        """Create a Gaussian potential well"""
        if self.dimensions == 1:
            return lambda x: -depth * np.exp(-(x - x0)**2 / (2 * width**2))
        else:
            return lambda x, y: -depth * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * width**2))

    def double_gaussian_well(self, x1, y1, x2, y2, depth, width):
        """Create a double Gaussian potential well"""
        if self.dimensions == 1:
            return lambda x: (-depth * np.exp(-(x - x1)**2 / (2 * width**2)) 
                            -depth * np.exp(-(x - x2)**2 / (2 * width**2)))
        else:
            return lambda x, y: (-depth * np.exp(-((x - x1)**2 + (y - y1)**2) / (2 * width**2))
                               -depth * np.exp(-((x - x2)**2 + (y - y2)**2) / (2 * width**2)))