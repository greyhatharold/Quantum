import numpy as np

class InitialStateFactory:
    @staticmethod
    def gaussian_wave_packet(solver, x0, p0, sigma):
        """
        Create a normalized Gaussian wave packet
        """
        initial_state = np.exp(-(solver.x - x0)**2 / (4 * sigma**2) + 1j * p0 * solver.x)
        return initial_state / np.sqrt(np.sum(np.abs(initial_state)**2) * solver.dx) 