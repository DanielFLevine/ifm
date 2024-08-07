import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdyn.datasets import generate_moons

# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n, var):
    return eight_normal_sample(n, 2, scale=5, var=var).float()


class IFMdatasets:
    def __init__(self, batch_size, dataset_name, dim, gaussian_var = 1, checker_size = 2):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dim = dim
        self.gaussian_var = gaussian_var
        self.checker_size = checker_size
        print("Initializing {} dataset with batch size = {}".format(dataset_name, batch_size))

        if self.dataset_name == "8gaussians" and not (self.gaussian_var == 0.1):
            print("Base variance for the 8 Gaussians dataset is 0.1, but you set to {}".format(self.gaussian_var))
        if self.dataset_name == "checkerboard":
            print("Checkboard configuration: {}-by-{}".format(checker_size, checker_size))
    
    def _shuffle(self, data):
        shuffled_indices = torch.randperm(data.size(0))
        return data[shuffled_indices]

    def _generate_spiral(self, noise=0.75, density_factor=1.5):
        """
        Generate a clear single 2D spiral dataset.

        Parameters:
        - n_points: The total number of points in the spiral
        - noise: The standard deviation of the Gaussian noise added to the data
        - density_factor: Factor to control the density of the spiral (higher values result in a more dense spiral)

        Returns:
        - X: A 2D array of shape (n_points, 2) containing the data points
        """
        # Generate angles linearly spaced between 0 and density_factor * 4π
        angles = np.linspace(0, density_factor * 4 * np.pi, self.batch_size)

        # Calculate x and y coordinates
        r = angles * density_factor
        x = r * np.cos(angles) + np.random.randn(self.batch_size) * noise
        y = r * np.sin(angles) + np.random.randn(self.batch_size) * noise

        X = np.vstack((x, y)).T
        X = torch.tensor(X, dtype=torch.float)

        return self._shuffle(X)
    
    def _generate_gaussian(self):
        m = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.dim), math.sqrt(self.gaussian_var) * torch.eye(self.dim))
        return m.sample((self.batch_size,))

    def _generate_checkerboard(self, reverse_checkerboard=False):
        '''
            This function generates a checkerboard dataset centered at the origin.
        '''
        size = self.checker_size
        num_squares = (size * size) // 2
        points_per_square = self.batch_size // num_squares

        # Generate coordinates for all squares
        # These are grid points such as (0,0), (0,1) etc. 
        # We will select a half of these to form the basis of a checkboard
        # Finally, we will add Uniform(0,1) noise to them to create checkerboard
        x_coords = np.repeat(np.arange(size), size)
        y_coords = np.tile(np.arange(size), size)

        # Determine squares to fill
        if not reverse_checkerboard:
            mask = (x_coords + y_coords) % 2 == 0
        else:
            mask = (x_coords + y_coords) % 2 != 0

        # Allocate points uniformly across to the bottom-left corner point of each square
        selected_x_coords = np.repeat(x_coords[mask], points_per_square)
        selected_y_coords = np.repeat(y_coords[mask], points_per_square)

        # Generate random offsets within each square
        offsets_x = np.random.rand(selected_x_coords.size)
        offsets_y = np.random.rand(selected_y_coords.size)

        # Calculate final point positions and center the pattern at the origin
        points_x = selected_x_coords + offsets_x - size / 2
        points_y = selected_y_coords + offsets_y - size / 2

        data = np.vstack((points_x, points_y)).T
        data = data * 2 # scale the data to spread out
        return self._shuffle(torch.tensor(data, dtype=torch.float))

    def _generate_gaussian_checkerboard(self, reverse_checkerboard=False):
        """
            Generates a checkerboard pattern centered at the origin.
        
            Parameters:
            - size: Size of the checkerboard (number of squares per row/column).
            - reverse_checkerboard: Whether to reverse the checkerboard pattern.
        """
        # Ensure batch_size is divisible by the number of squares to be filled
        size = self.checker_size
        num_squares = size * size // 2
        points_per_square = self.batch_size // num_squares
        adjusted_batch_size = points_per_square * num_squares

        # Generate coordinates for the checkerboard
        x_coords = np.repeat(np.arange(size), size)
        y_coords = np.tile(np.arange(size), size)

        # Determine squares to fill
        mask = (x_coords + y_coords) % 2 == 0 if not reverse_checkerboard else (x_coords + y_coords) % 2 != 0

        # Allocate points uniformly across to the bottom-left corner point of each square
        selected_x_coords = np.repeat(x_coords[mask], points_per_square)
        selected_y_coords = np.repeat(y_coords[mask], points_per_square)

        # Generate random noise to offset points within each square
        noise = np.random.normal(loc=0.0, scale=math.sqrt(self.gaussian_var), size=(adjusted_batch_size, self.dim))

        # Calculate final point positions
        points_x = selected_x_coords + noise[:, 0]
        points_y = selected_y_coords + noise[:, 1]

        # Centering at the origin and scaling
        data = np.vstack((points_x, points_y)).T
        data = (data - np.mean(data, axis=0)) * 4

        # Shuffle the data and return
        return self._shuffle(torch.tensor(data, dtype=torch.float))

    
    def _generate_8gaussians(self):

        return sample_8gaussians(self.batch_size, var=self.gaussian_var)
    
    def _generate_moons(self):
        return self._shuffle(sample_moons(self.batch_size))

    def _generate_4gaussian_blobs(self, spacing=10, cov_scale=1.0):
        """
        Generate samples from Gaussian blobs with means symmetrically placed along the first dimension.
        
        Parameters:
        - n_blobs: int, number of Gaussian blobs (expected to be 3 for this specific setup).
        - points_per_blob: int, number of points per Gaussian blob.
        - spacing: float, spacing between the blobs along the first dimension.
        - cov_scale: float, scaling factor for the identity covariance matrix.
        
        Returns:
        - torch.Tensor: Concatenated samples from all Gaussian blobs.
        """
        # Ensure the configuration is as expected
        n_blobs=4
        num_points = self.batch_size
        points_per_blob = int(num_points/n_blobs)
        dimension = self.dim
        
        # Define the means for the blobs, symmetrically around 0 along the first dimension
        means = torch.tensor([[-1.5*spacing, 0], [-0.5*spacing, 0], [0.5*spacing, 0], [1.5*spacing, 0]]).float()
        means = torch.cat((means, torch.zeros(n_blobs, dimension - 2)), dim=1)  # Extend means to high-dimensional space

        # Isotropic covariance matrix scaled by cov_scale
        cov_target = torch.eye(dimension) * cov_scale

        # Sample points from each blob
        target_data_list = []
        for mean in means:
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_target)
            samples = dist.sample((points_per_blob,))
            target_data_list.append(samples)

        # Concatenate all samples
        target_data = torch.cat(target_data_list, dim=0)

        return target_data

    def _plot_sample(self):
        sample = self.generate_data()
        plt.scatter(sample[:, 0], sample[:, 1], s = 1)
        plt.show()
        plt.close()
    
    
    def generate_data(self, batch_size=None):
        if batch_size is not None: # override batchsize
            self.batch_size = batch_size
            
        if self.dataset_name == "gaussian":
            return self._generate_gaussian()
        elif self.dataset_name == "spiral":
            return self._generate_spiral()
        elif self.dataset_name == "checkerboard":
            return self._generate_checkerboard()
        elif self.dataset_name == "8gaussians":
            return self._generate_8gaussians()
        elif self.dataset_name == "2moons":
            return self._generate_moons()
        elif self.dataset_name == "4blobs":
            return self._generate_4gaussian_blobs()
        elif self.dataset_name == "gaussian_checkerboard":
            return self._generate_gaussian_checkerboard()
        else:
            raise NotImplementedError



