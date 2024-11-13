import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from core.quantum_solver import QuantumSolver

class QuantumGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Solver GUI")
        
        # Initialize solver with default values
        self.solver = QuantumSolver()
        self.setup_default_values()
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create visualization mode selector
        self.create_mode_selector()
        
        # Create main content area with matplotlib
        self.create_main_content()
        
        # Create sidebar
        self.sidebar_visible = False
        self.create_sidebar()
        
        # Create toggle button
        self.create_toggle_button()
        
        # Initial plot
        self.update_simulation()

    def setup_default_values(self):
        self.params = {
            'x_min': tk.DoubleVar(value=-10),
            'x_max': tk.DoubleVar(value=10),
            'y_min': tk.DoubleVar(value=-10),
            'y_max': tk.DoubleVar(value=10),
            'n_points_x': tk.IntVar(value=50),
            'n_points_y': tk.IntVar(value=50),
            'n_states': tk.IntVar(value=5),
            'omega': tk.DoubleVar(value=1.0),
            'x0': tk.DoubleVar(value=-2.0),
            'y0': tk.DoubleVar(value=0.0),
            'px0': tk.DoubleVar(value=1.0),
            'py0': tk.DoubleVar(value=0.0),
            'sigma': tk.DoubleVar(value=0.5),
            't_max': tk.DoubleVar(value=10),
            't_steps': tk.IntVar(value=200),
            'animation_interval': tk.IntVar(value=50),
            'well_depth': tk.DoubleVar(value=5.0),
            'well_width': tk.DoubleVar(value=1.0),
            'well_x1': tk.DoubleVar(value=-2.0),
            'well_y1': tk.DoubleVar(value=0.0),
            'well_x2': tk.DoubleVar(value=2.0),
            'well_y2': tk.DoubleVar(value=0.0),
            'double_well': tk.BooleanVar(value=False)
        }
        self.visualization_mode = tk.StringVar(value='2d')

    def create_mode_selector(self):
        mode_frame = ttk.Frame(self.main_container)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="1D", variable=self.visualization_mode,
                       value='2d', command=self.switch_visualization).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="2D", variable=self.visualization_mode,
                       value='3d', command=self.switch_visualization).pack(side=tk.LEFT)

    def create_sidebar(self):
        self.sidebar = ttk.Frame(self.main_container, padding="10")
        
        # Create input fields with scales
        params_config = {
            'x_min': (-20, 0),
            'x_max': (0, 20),
            'y_min': (-20, 0),
            'y_max': (0, 20),
            'n_points_x': (20, 100),
            'n_points_y': (20, 100),
            'n_states': (1, 10),
            'omega': (0.1, 5.0),
            'x0': (-5, 5),
            'y0': (-5, 5),
            'px0': (-5, 5),
            'py0': (-5, 5),
            'sigma': (0.1, 2.0),
            't_max': (1, 20),
            't_steps': (50, 500),
            'animation_interval': (10, 200),
            'well_depth': (0.1, 10.0),
            'well_width': (0.1, 5.0),
            'well_x1': (-5, 5),
            'well_y1': (-5, 5),
            'well_x2': (-5, 5),
            'well_y2': (-5, 5)
        }
        
        # Create frames for 1D and 2D parameters
        self.params_1d = ttk.Frame(self.sidebar)
        self.params_2d = ttk.Frame(self.sidebar)
        
        # Add parameters to appropriate frames
        for param, (min_val, max_val) in params_config.items():
            if param.startswith('y'):
                frame = self.params_2d
            else:
                frame = self.params_1d
                
            param_frame = ttk.Frame(frame)
            param_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(param_frame, text=param).pack(fill=tk.X)
            
            entry = ttk.Entry(param_frame, textvariable=self.params[param])
            entry.pack(fill=tk.X)
            
            scale = ttk.Scale(param_frame, from_=min_val, to=max_val, 
                            variable=self.params[param],
                            orient=tk.HORIZONTAL)
            scale.pack(fill=tk.X)
        
        # Add Gaussian well parameters
        well_frame = ttk.LabelFrame(self.sidebar, text="Gaussian Well Parameters")
        well_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(well_frame, text="Double Well", 
                       variable=self.params['double_well']).pack()
        
        ttk.Button(self.sidebar, text="Update", 
                  command=self.update_simulation).pack(pady=10)
        
        # Show appropriate parameter frame
        self.switch_visualization()

    def switch_visualization(self):
        # Hide both frames
        self.params_1d.pack_forget()
        self.params_2d.pack_forget()
        
        # Show appropriate frame
        if self.visualization_mode.get() == '2d':
            self.params_1d.pack(fill=tk.BOTH, expand=True)
        else:
            self.params_2d.pack(fill=tk.BOTH, expand=True)
        
        self.update_simulation()

    def create_main_content(self):
        """Create matplotlib canvas for both 1D and 2D visualizations"""
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_container)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.main_container)
        toolbar.update()

    def update_simulation(self):
        """Update visualization based on current mode"""
        if self.visualization_mode.get() == '2d':
            self.update_1d_simulation()
        else:
            self.update_2d_simulation()

    def update_1d_simulation(self):
        """Update 1D simulation visualization"""
        self.solver = QuantumSolver(
            x_min=self.params['x_min'].get(),
            x_max=self.params['x_max'].get(),
            n_points_x=self.params['n_points_x'].get()
        )
        
        # Set 1D potential (harmonic oscillator)
        omega = self.params['omega'].get()
        V = lambda x: 0.5 * self.solver.m * omega**2 * x**2
        self.solver.set_potential(V)
        
        # Solve and visualize
        self.solver.solve_stationary_states(n_states=self.params['n_states'].get())
        self.fig.clear()
        self.solver.plot_animation_frame(self.fig, self.solver.eigenstates[:, 0])
        self.canvas.draw()

    def update_2d_simulation(self):
        """Update 2D simulation visualization using matplotlib"""
        self.solver = QuantumSolver(
            x_min=self.params['x_min'].get(),
            x_max=self.params['x_max'].get(),
            y_min=self.params['y_min'].get(),
            y_max=self.params['y_max'].get(),
            n_points_x=self.params['n_points_x'].get(),
            n_points_y=self.params['n_points_y'].get()
        )
        
        # Set up Gaussian well potential
        if self.params['double_well'].get():
            V = self.solver.double_gaussian_well(
                self.params['well_x1'].get(),
                self.params['well_y1'].get(),
                self.params['well_x2'].get(),
                self.params['well_y2'].get(),
                self.params['well_depth'].get(),
                self.params['well_width'].get()
            )
        else:
            V = self.solver.gaussian_well(
                self.params['well_x1'].get(),
                self.params['well_y1'].get(),
                self.params['well_depth'].get(),
                self.params['well_width'].get()
            )
            
        self.solver.set_potential(V)
        
        # Solve and visualize
        self.solver.solve_stationary_states(n_states=self.params['n_states'].get())
        
        # Clear previous plot
        self.fig.clear()
        
        # Create new 3D plot
        ax = self.fig.add_subplot(111, projection='3d')
        state_2d = self.solver.eigenstates[:, 0].reshape(len(self.solver.y), len(self.solver.x))
        probability = np.abs(state_2d)**2
        
        surf = ax.plot_surface(self.solver.X, self.solver.Y, probability,
                             cmap='viridis', antialiased=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        self.fig.colorbar(surf)
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def create_toggle_button(self):
        """Create button to toggle sidebar visibility"""
        toggle_btn = ttk.Button(self.main_container, text="Toggle Parameters",
                              command=self.toggle_sidebar)
        toggle_btn.pack(side=tk.TOP, pady=5)

    def toggle_sidebar(self):
        """Toggle sidebar visibility"""
        if self.sidebar_visible:
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_visible = not self.sidebar_visible
        