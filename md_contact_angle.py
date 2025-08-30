"""
Molecular Dynamics simulation of contact angle formation.
Based on: https://pubs.aip.org/aip/jcp/article/130/3/034705/350620/Molecular-dynamics-simulation-of-the-contact-angle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

class MDContactAngle:
    def __init__(self, N_liquid=1000, N_solid=500, box_size=20.0, T=0.7):
        # System parameters
        self.N_liquid = N_liquid  # Number of liquid particles
        self.N_solid = N_solid    # Number of solid particles
        self.box_size = box_size  # Simulation box size
        self.T = T                # Reduced temperature
        
        # Lennard-Jones parameters
        self.epsilon = 1.0        # Energy scale
        self.sigma = 1.0          # Length scale
        self.rcut = 2.5 * self.sigma  # Cutoff radius
        
        # Solid-liquid interaction parameters
        self.epsilon_sl = 1.0     # Solid-liquid interaction strength
        self.sigma_sl = 1.0       # Solid-liquid interaction length
        
        # Initialize positions
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities()
        self.forces = np.zeros_like(self.positions)
        
        # Time step
        self.dt = 0.005
        
        # Storage for analysis
        self.contact_angles = []
        self.times = []
        
        # Debug counters
        self.force_calls = 0
        self.angle_measurements = 0
        
        # Visualization setup
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.scatter = None
        self.angle_line = None
        self.contact_angle_text = None
        
    def initialize_positions(self):
        """Initialize particle positions"""
        # Create solid surface (bottom layer)
        solid_pos = np.zeros((self.N_solid, 3))
        n_solid_x = int(np.sqrt(self.N_solid))
        spacing = self.box_size / n_solid_x
        
        for i in range(n_solid_x):
            for j in range(n_solid_x):
                idx = i * n_solid_x + j
                if idx < self.N_solid:
                    solid_pos[idx] = [i * spacing, j * spacing, 0.0]
        
        # Create liquid droplet (spherical)
        liquid_pos = np.zeros((self.N_liquid, 3))
        center = np.array([self.box_size/2, self.box_size/2, self.box_size/4])  # Start closer to surface
        radius = 6.0  # Increased radius
        
        # More efficient droplet initialization using grid
        grid_size = int(np.ceil(np.cbrt(self.N_liquid * 2)))  # Increased grid size
        spacing = 2 * radius / grid_size
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if idx >= self.N_liquid:
                        break
                    # Create grid point
                    x = center[0] - radius + i * spacing
                    y = center[1] - radius + j * spacing
                    z = center[2] - radius + k * spacing
                    
                    # Check if point is within sphere
                    if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                        liquid_pos[idx] = [x, y, z]
                        idx += 1
        
        if idx < self.N_liquid:
            logger.warning(f"Could not place all liquid particles. Placed {idx}/{self.N_liquid}")
            liquid_pos = liquid_pos[:idx]
            self.N_liquid = idx
        
        return np.vstack((solid_pos, liquid_pos))
    
    def initialize_velocities(self):
        """Initialize velocities with Maxwell-Boltzmann distribution"""
        velocities = np.random.normal(0, np.sqrt(self.T), (self.N_liquid + self.N_solid, 3))
        # Remove center of mass velocity
        velocities -= np.mean(velocities, axis=0)
        return velocities
    
    def lj_potential(self, r):
        """Lennard-Jones potential"""
        if r < 1e-10:  # Prevent division by zero
            return float('inf')
        r6 = (self.sigma/r)**6
        r12 = r6**2
        return 4 * self.epsilon * (r12 - r6)
    
    def lj_force(self, r):
        """Lennard-Jones force"""
        if r < 1e-10:  # Prevent division by zero
            return 0.0
        r6 = (self.sigma/r)**6
        r12 = r6**2
        return 24 * self.epsilon * (2*r12 - r6) / r
    
    def calculate_forces(self):
        """Calculate forces between all particles"""
        self.force_calls += 1
        self.forces = np.zeros_like(self.positions)
        
        try:
            # Calculate forces between liquid particles
            for i in range(self.N_solid, self.N_solid + self.N_liquid):
                for j in range(i + 1, self.N_solid + self.N_liquid):
                    r_ij = self.positions[j] - self.positions[i]
                    r_ij = self.apply_pbc(r_ij)
                    r = np.linalg.norm(r_ij)
                    
                    if r < self.rcut and r > 1e-10:
                        force = self.lj_force(r) * r_ij/r
                        self.forces[i] += force
                        self.forces[j] -= force
            
            # Calculate forces between liquid and solid particles
            for i in range(self.N_solid, self.N_solid + self.N_liquid):
                for j in range(self.N_solid):
                    r_ij = self.positions[j] - self.positions[i]
                    r_ij = self.apply_pbc(r_ij)
                    r = np.linalg.norm(r_ij)
                    
                    if r < self.rcut and r > 1e-10:
                        force = self.lj_force(r) * r_ij/r
                        self.forces[i] += force
                        self.forces[j] -= force
                        
        except Exception as e:
            logger.error(f"Error in force calculation: {e}")
            logger.error(f"Force calculation attempt {self.force_calls}")
            raise
    
    def apply_pbc(self, r):
        """Apply periodic boundary conditions"""
        return r - np.round(r/self.box_size) * self.box_size
    
    def measure_contact_angle(self):
        """Measure contact angle using density profile and interface fitting"""
        self.angle_measurements += 1
        try:
            # Calculate density profile in x-z plane
            x_bins = np.linspace(0, self.box_size, 100)
            z_bins = np.linspace(0, self.box_size, 100)
            rho = np.zeros((len(x_bins), len(z_bins)))
            
            for i in range(self.N_solid, self.N_solid + self.N_liquid):
                x = self.positions[i, 0]
                z = self.positions[i, 2]
                x_idx = np.digitize(x, x_bins) - 1
                z_idx = np.digitize(z, z_bins) - 1
                if 0 <= x_idx < len(x_bins) and 0 <= z_idx < len(z_bins):
                    rho[x_idx, z_idx] += 1
            
            # Find interface points
            interface_points = []
            for i in range(len(x_bins)):
                for j in range(len(z_bins)):
                    if rho[i, j] > 0:
                        # Check if it's an interface point
                        if (i > 0 and rho[i-1, j] == 0) or (i < len(x_bins)-1 and rho[i+1, j] == 0) or \
                           (j > 0 and rho[i, j-1] == 0) or (j < len(z_bins)-1 and rho[i, j+1] == 0):
                            interface_points.append((x_bins[i], z_bins[j]))
            
            if len(interface_points) < 3:
                return None, None
            
            # Convert to numpy array
            interface_points = np.array(interface_points)
            x = interface_points[:, 0]
            z = interface_points[:, 1]
            
            # Fit a line to the interface points near the surface
            surface_threshold = self.box_size/4  # Points near the surface
            surface_points = interface_points[z < surface_threshold]
            
            if len(surface_points) < 2:
                return None, None
            
            # Fit a line to the surface points
            def line(x, a, b):
                return a * x + b
            
            try:
                popt, _ = curve_fit(line, surface_points[:, 0], surface_points[:, 1])
                slope = popt[0]
                intercept = popt[1]
                
                # Calculate contact angle from slope
                contact_angle = np.arctan(np.abs(slope)) * 180/np.pi
                return contact_angle, (slope, intercept)
                
            except Exception as e:
                logger.warning(f"Line fitting failed: {e}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in contact angle measurement: {e}")
            logger.error(f"Measurement attempt {self.angle_measurements}")
            return None, None
    
    def update_visualization(self):
        """Update the visualization"""
        # Clear previous plot
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot particles
        solid_x = self.positions[:self.N_solid, 0]
        solid_z = self.positions[:self.N_solid, 2]
        liquid_x = self.positions[self.N_solid:, 0]
        liquid_z = self.positions[self.N_solid:, 2]
        
        # Plot solid particles
        self.ax1.scatter(solid_x, solid_z, c='gray', s=10, alpha=0.5)
        # Plot liquid particles
        self.ax1.scatter(liquid_x, liquid_z, c='blue', s=10, alpha=0.5)
        
        # Set plot limits
        self.ax1.set_xlim(0, self.box_size)
        self.ax1.set_ylim(0, self.box_size)
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('z')
        self.ax1.set_title('Particle Positions')
        
        # Plot contact angle
        angle, line_params = self.measure_contact_angle()
        if angle is not None:
            slope, intercept = line_params
            x_line = np.linspace(0, self.box_size, 100)
            z_line = slope * x_line + intercept
            self.ax1.plot(x_line, z_line, 'r-', linewidth=2)
            
            # Add contact angle text
            self.ax1.text(0.05, 0.95, f'Contact Angle: {angle:.1f}°',
                         transform=self.ax1.transAxes, fontsize=12,
                         verticalalignment='top')
        
        # Plot contact angle evolution
        if len(self.contact_angles) > 0:
            self.ax2.plot(self.times, self.contact_angles, 'b-')
            self.ax2.set_xlabel('Time step')
            self.ax2.set_ylabel('Contact angle (degrees)')
            self.ax2.set_title('Contact Angle Evolution')
            self.ax2.grid(True)
        
        # Draw the plot
        plt.draw()
        plt.pause(0.001)
    
    def step(self):
        """Advance simulation by one time step"""
        try:
            # Update positions
            self.positions += self.velocities * self.dt + 0.5 * self.forces * self.dt**2
            
            # Apply periodic boundary conditions
            self.positions = self.positions % self.box_size
            
            # Calculate new forces
            old_forces = self.forces.copy()
            self.calculate_forces()
            
            # Update velocities
            self.velocities += 0.5 * (old_forces + self.forces) * self.dt
            
            # Measure contact angle
            angle, _ = self.measure_contact_angle()
            if angle is not None:
                self.contact_angles.append(angle)
                self.times.append(len(self.times))
            
            # Update visualization
            self.update_visualization()
                
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            raise
    
    def run(self, n_steps=1000):
        """Run simulation for specified number of steps"""
        try:
            # Initial visualization
            self.update_visualization()
            
            for step in range(n_steps):
                self.step()
                
                # Print progress
                if step % 100 == 0:
                    logger.info(f"Step {step}/{n_steps}")
                    if len(self.contact_angles) > 0:
                        logger.info(f"Current contact angle: {self.contact_angles[-1]:.2f}°")
                    logger.info(f"Force calculations: {self.force_calls}")
                    logger.info(f"Angle measurements: {self.angle_measurements}")
                    
        except Exception as e:
            logger.error(f"Simulation failed at step {step}: {e}")
            raise
        finally:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep the final plot open

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create simulation
        sim = MDContactAngle(N_liquid=500, N_solid=200)  # Reduced number of particles for testing
        
        # Run simulation
        sim.run(n_steps=1000)
        
    except Exception as e:
        logger.error(f"Main simulation failed: {e}")

if __name__ == '__main__':
    main() 