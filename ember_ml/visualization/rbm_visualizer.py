"""
RBM Visualization Module

This module provides visualization tools for Restricted Boltzmann Machines,
including static plots and animations that showcase the learning process
and the "dreaming" capabilities of RBMs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import time

# Import the RBM class
from ember_ml.models.rbm import RestrictedBoltzmannMachine


class RBMVisualizer:
    """
    Visualization tools for Restricted Boltzmann Machines.
    
    This class provides methods for creating static plots and animations
    that showcase the learning process and generative capabilities of RBMs.
    Visualizations include:
    
    - Weight matrices and their evolution during training
    - Hidden unit activations
    - Reconstruction quality
    - "Dreaming" sequences showing the RBM's generative capabilities
    - Anomaly detection visualizations
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs',
        plots_dir: str = 'plots',
        animations_dir: str = 'animations',
        dpi: int = 100,
        cmap: str = 'viridis',
        figsize: Tuple[int, int] = (10, 8),
        animation_interval: int = 200
    ):
        """
        Initialize the RBM visualizer.
        
        Args:
            output_dir: Base output directory
            plots_dir: Directory for static plots (relative to output_dir)
            animations_dir: Directory for animations (relative to output_dir)
            dpi: DPI for saved figures
            cmap: Colormap for plots
            figsize: Default figure size
            animation_interval: Default interval between animation frames (ms)
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, plots_dir)
        self.animations_dir = os.path.join(output_dir, animations_dir)
        self.dpi = dpi
        self.cmap = cmap
        self.figsize = figsize
        self.animation_interval = animation_interval
        
        # Create output directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.animations_dir, exist_ok=True)
        
        # Create a custom colormap for weight visualization
        # This creates a diverging colormap with white at the center
        self.weight_cmap = LinearSegmentedColormap.from_list(
            'weight_cmap',
            ['#3b4cc0', '#white', '#b40426']
        )
    
    def plot_training_curve(
        self,
        rbm: RestrictedBoltzmannMachine,
        title: str = 'RBM Training Curve',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the training curve (reconstruction error vs. epoch).
        
        Args:
            rbm: Trained RBM
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot training errors
        ax.plot(rbm.training_errors, 'b-', linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reconstruction Error', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text with training information
        info_text = (
            f"Visible units: {rbm.n_visible}\n"
            f"Hidden units: {rbm.n_hidden}\n"
            f"Learning rate: {rbm.learning_rate}\n"
            f"Final error: {rbm.training_errors[-1]:.4f}\n"
            f"Training time: {rbm.training_time:.2f}s"
        )
        ax.text(
            0.02, 0.95, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_training_curve_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Training curve saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_weight_matrix(
        self,
        rbm: RestrictedBoltzmannMachine,
        reshape_visible: Optional[Tuple[int, int]] = None,
        reshape_hidden: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Weight Matrix',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the weight matrix of the RBM.
        
        Args:
            rbm: Trained RBM
            reshape_visible: Optional shape to reshape visible units (for images)
            reshape_hidden: Optional shape to reshape hidden units
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Determine if we should use a grid layout
        use_grid = reshape_visible is not None and reshape_hidden is not None
        
        if use_grid:
            # Create a grid of weight visualizations for image data
            n_vis_rows, n_vis_cols = reshape_visible
            n_hid_rows, n_hid_cols = reshape_hidden
            
            fig, axes = plt.subplots(
                n_hid_rows, n_hid_cols,
                figsize=(n_hid_cols * 2, n_hid_rows * 2)
            )
            axes = axes.flatten()
            
            for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                # Reshape weights for this hidden unit into an image
                weight_img = rbm.weights[:, h].reshape(n_vis_rows, n_vis_cols)
                
                # Plot the weight image
                im = axes[h].imshow(
                    weight_img,
                    cmap=self.weight_cmap,
                    interpolation='nearest'
                )
                axes[h].set_title(f"H{h+1}")
                axes[h].axis('off')
            
            # Hide unused axes
            for h in range(rbm.n_hidden, len(axes)):
                axes[h].axis('off')
            
            # Add a colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            plt.suptitle(title, fontsize=16)
            
        else:
            # Create a heatmap of the full weight matrix
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot the weight matrix
            im = ax.imshow(
                rbm.weights,
                cmap=self.cmap,
                aspect='auto',
                interpolation='nearest'
            )
            
            # Add a colorbar
            plt.colorbar(im, ax=ax)
            
            # Add labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Hidden Units', fontsize=12)
            ax.set_ylabel('Visible Units', fontsize=12)
            
            # Add grid lines if the matrix is not too large
            if rbm.n_visible < 50 and rbm.n_hidden < 50:
                ax.set_xticks(np.arange(rbm.n_hidden))
                ax.set_yticks(np.arange(rbm.n_visible))
                ax.grid(False)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_weight_matrix_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Weight matrix plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_reconstructions(
        self,
        rbm: RestrictedBoltzmannMachine,
        data: np.ndarray,
        n_samples: int = 5,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Reconstructions',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot original data samples and their reconstructions.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to plot
            reshape: Optional shape to reshape samples (for images)
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Select random samples
        indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Reconstruct samples
        reconstructions = rbm.reconstruct(samples)
        
        # Create figure
        fig, axes = plt.subplots(
            n_samples, 2,
            figsize=(6, n_samples * 3)
        )
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = np.array([axes])
        
        # Plot original and reconstructed samples
        for i in range(n_samples):
            # Original sample
            if reshape is not None:
                # Reshape for image data
                orig_img = samples[i].reshape(reshape)
                recon_img = reconstructions[i].reshape(reshape)
                
                axes[i, 0].imshow(orig_img, cmap='gray', interpolation='nearest')
                axes[i, 1].imshow(recon_img, cmap='gray', interpolation='nearest')
            else:
                # Bar plot for non-image data
                axes[i, 0].bar(range(len(samples[i])), samples[i])
                axes[i, 1].bar(range(len(reconstructions[i])), reconstructions[i])
                
                # Set y-axis limits
                y_min = min(samples[i].min(), reconstructions[i].min())
                y_max = max(samples[i].max(), reconstructions[i].max())
                axes[i, 0].set_ylim(y_min, y_max)
                axes[i, 1].set_ylim(y_min, y_max)
            
            # Set titles and turn off axis labels
            if i == 0:
                axes[i, 0].set_title('Original')
                axes[i, 1].set_title('Reconstructed')
            
            axes[i, 0].set_xticks([])
            axes[i, 1].set_xticks([])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_reconstructions_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Reconstructions plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_hidden_activations(
        self,
        rbm: RestrictedBoltzmannMachine,
        data: np.ndarray,
        n_samples: int = 5,
        n_hidden_units: int = 20,
        title: str = 'RBM Hidden Unit Activations',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot hidden unit activations for data samples.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to plot
            n_hidden_units: Number of hidden units to plot
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Select random samples
        indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Compute hidden activations
        hidden_probs = rbm.compute_hidden_probabilities(samples)
        
        # Limit number of hidden units to plot
        n_hidden_units = min(n_hidden_units, rbm.n_hidden)
        hidden_probs = hidden_probs[:, :n_hidden_units]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            hidden_probs,
            cmap=self.cmap,
            aspect='auto',
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hidden Unit', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        
        # Add grid lines
        ax.set_xticks(np.arange(n_hidden_units))
        ax.set_yticks(np.arange(n_samples))
        ax.set_xticklabels([f"H{i+1}" for i in range(n_hidden_units)])
        ax.set_yticklabels([f"S{i+1}" for i in range(n_samples)])
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_hidden_activations_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Hidden activations plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_anomaly_scores(
        self,
        rbm: RestrictedBoltzmannMachine,
        normal_data: np.ndarray,
        anomaly_data: Optional[np.ndarray] = None,
        method: str = 'reconstruction',
        title: str = 'RBM Anomaly Scores',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot anomaly scores for normal and anomalous data.
        
        Args:
            rbm: Trained RBM
            normal_data: Normal data
            anomaly_data: Anomalous data (optional)
            method: Method to use ('reconstruction' or 'free_energy')
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Compute anomaly scores
        normal_scores = rbm.anomaly_score(normal_data, method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal scores
        ax.hist(
            normal_scores,
            bins=30,
            alpha=0.7,
            color='blue',
            label='Normal'
        )
        
        # Plot anomaly scores if provided
        if anomaly_data is not None:
            anomaly_scores = rbm.anomaly_score(anomaly_data, method)
            ax.hist(
                anomaly_scores,
                bins=30,
                alpha=0.7,
                color='red',
                label='Anomaly'
            )
        
        # Add threshold line
        if method == 'reconstruction':
            threshold = rbm.reconstruction_error_threshold
            threshold_label = 'Reconstruction Error Threshold'
        else:
            threshold = rbm.free_energy_threshold
            threshold_label = 'Free Energy Threshold'
        
        if threshold is not None:
            ax.axvline(
                threshold,
                color='black',
                linestyle='--',
                linewidth=2,
                label=threshold_label
            )
        
        # Add labels and title
        if method == 'reconstruction':
            ax.set_xlabel('Reconstruction Error', fontsize=12)
        else:
            ax.set_xlabel('Free Energy', fontsize=12)
            
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_anomaly_scores_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Anomaly scores plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def animate_weight_evolution(
        self,
        rbm: RestrictedBoltzmannMachine,
        reshape_visible: Optional[Tuple[int, int]] = None,
        reshape_hidden: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Weight Evolution',
        interval: int = 200,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the evolution of weights during training.
        
        Args:
            rbm: Trained RBM with training_states
            reshape_visible: Optional shape to reshape visible units (for images)
            reshape_hidden: Optional shape to reshape hidden units
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        if not rbm.training_states:
            raise ValueError("No training states available. Train the RBM with track_states=True.")
        
        # Determine if we should use a grid layout
        use_grid = reshape_visible is not None and reshape_hidden is not None
        
        if use_grid:
            # Create a grid of weight visualizations for image data
            n_vis_rows, n_vis_cols = reshape_visible
            n_hid_rows, n_hid_cols = reshape_hidden
            
            fig, axes = plt.subplots(
                n_hid_rows, n_hid_cols,
                figsize=(n_hid_cols * 2, n_hid_rows * 2)
            )
            axes = axes.flatten()
            
            # Initialize images
            images = []
            for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                # Reshape weights for this hidden unit into an image
                weight_img = rbm.training_states[0]['weights'][:, h].reshape(n_vis_rows, n_vis_cols)
                
                # Plot the weight image
                im = axes[h].imshow(
                    weight_img,
                    cmap=self.weight_cmap,
                    interpolation='nearest',
                    animated=True
                )
                axes[h].set_title(f"H{h+1}")
                axes[h].axis('off')
                images.append(im)
            
            # Hide unused axes
            for h in range(rbm.n_hidden, len(axes)):
                axes[h].axis('off')
            
            # Add a colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(images[0], cax=cbar_ax)
            
            # Add error text
            error_text = fig.text(
                0.5, 0.01,
                f"Epoch: 0, Error: {rbm.training_states[0]['error']:.4f}",
                ha='center',
                fontsize=12
            )
            
            plt.suptitle(title, fontsize=16)
            
            # Animation update function
            def update(frame):
                state = rbm.training_states[frame]
                
                for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                    weight_img = state['weights'][:, h].reshape(n_vis_rows, n_vis_cols)
                    images[h].set_array(weight_img)
                
                error_text.set_text(f"Epoch: {frame}, Error: {state['error']:.4f}")
                
                return images + [error_text]
            
        else:
            # Create a heatmap of the full weight matrix
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot the initial weight matrix
            im = ax.imshow(
                rbm.training_states[0]['weights'],
                cmap=self.cmap,
                aspect='auto',
                interpolation='nearest',
                animated=True
            )
            
            # Add a colorbar
            plt.colorbar(im, ax=ax)
            
            # Add labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Hidden Units', fontsize=12)
            ax.set_ylabel('Visible Units', fontsize=12)
            
            # Add error text
            error_text = ax.text(
                0.02, 0.95,
                f"Epoch: 0, Error: {rbm.training_states[0]['error']:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Animation update function
            def update(frame):
                state = rbm.training_states[frame]
                im.set_array(state['weights'])
                error_text.set_text(f"Epoch: {frame}, Error: {state['error']:.4f}")
                return [im, error_text]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(rbm.training_states),
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_weight_evolution_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Weight evolution animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani
    
    def animate_dreaming(
        self,
        rbm: RestrictedBoltzmannMachine,
        n_steps: int = 100,
        start_data: Optional[np.ndarray] = None,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Dreaming',
        interval: int = 200,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the RBM "dreaming" process.
        
        Args:
            rbm: Trained RBM
            n_steps: Number of dreaming steps
            start_data: Optional starting data (if None, random initialization)
            reshape: Optional shape to reshape samples (for images)
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        # Generate dream states
        dream_states = rbm.dream(n_steps, start_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Initialize plot
        if reshape is not None:
            # Image data
            im = ax.imshow(
                dream_states[0].reshape(reshape),
                cmap='gray',
                interpolation='nearest',
                animated=True,
                vmin=0,
                vmax=1
            )
            ax.axis('off')
        else:
            # Non-image data
            bars = ax.bar(
                range(rbm.n_visible),
                dream_states[0].flatten(),
                animated=True
            )
            ax.set_ylim(0, 1)
            ax.set_xlabel('Visible Unit', fontsize=12)
            ax.set_ylabel('Activation', fontsize=12)
        
        # Add step counter
        step_text = ax.text(
            0.02, 0.95,
            f"Step: 0/{n_steps}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_title(title, fontsize=14)
        
        # Animation update function
        def update(frame):
            step_text.set_text(f"Step: {frame+1}/{n_steps}")
            
            if reshape is not None:
                # Update image
                im.set_array(dream_states[frame].reshape(reshape))
                return [im, step_text]
            else:
                # Update bars
                for i, bar in enumerate(bars):
                    bar.set_height(dream_states[frame].flatten()[i])
                return bars + [step_text]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(dream_states),
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_dreaming_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Dreaming animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani
    
    def animate_reconstruction(
        self,
        rbm: RestrictedBoltzmannMachine,
        data: np.ndarray,
        n_samples: int = 5,
        n_steps: int = 10,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Reconstruction Process',
        interval: int = 300,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the reconstruction process.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to animate
            n_steps: Number of Gibbs sampling steps
            reshape: Optional shape to reshape samples (for images)
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        # Select random samples
        indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Create figure
        fig, axes = plt.subplots(
            n_samples, 2,
            figsize=(8, n_samples * 3)
        )
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = np.array([axes])
        
        # Initialize plots
        images_orig = []
        images_recon = []
        
        for i in range(n_samples):
            # Original sample (left)
            if reshape is not None:
                # Image data
                im_orig = axes[i, 0].imshow(
                    samples[i].reshape(reshape),
                    cmap='gray',
                    interpolation='nearest',
                    animated=True,
                    vmin=0,
                    vmax=1
                )
                axes[i, 0].axis('off')
                
                # Initial reconstruction (right) - starts as copy of original
                im_recon = axes[i, 1].imshow(
                    samples[i].reshape(reshape),
                    cmap='gray',
                    interpolation='nearest',
                    animated=True,
                    vmin=0,
                    vmax=1
                )
                axes[i, 1].axis('off')
            else:
                # Non-image data
                im_orig = axes[i, 0].bar(
                    range(rbm.n_visible),
                    samples[i],
                    animated=True
                )
                
                im_recon = axes[i, 1].bar(
                    range(rbm.n_visible),
                    samples[i],
                    animated=True
                )
                
                # Set y-axis limits
                y_max = samples[i].max() * 1.1
                axes[i, 0].set_ylim(0, y_max)
                axes[i, 1].set_ylim(0, y_max)
            
            # Set titles for first row
            if i == 0:
                axes[i, 0].set_title('Original')
                axes[i, 1].set_title('Reconstruction')
            
            images_orig.append(im_orig)
            images_recon.append(im_recon)
        
        # Add step counter
        step_text = fig.text(
            0.5, 0.01,
            f"Step: 0/{n_steps}",
            ha='center',
            fontsize=12
        )
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle and step counter
        
        # Precompute reconstruction steps
        reconstruction_steps = []
        current_samples = samples.copy()
        
        for step in range(n_steps):
            # Compute hidden probabilities and sample states
            hidden_probs = rbm.compute_hidden_probabilities(current_samples)
            hidden_states = rbm.sample_hidden_states(hidden_probs)
            
            # Compute visible probabilities and sample states
            visible_probs = rbm.compute_visible_probabilities(hidden_states)
            visible_states = rbm.sample_visible_states(visible_probs)
            
            # Store reconstructed samples
            reconstruction_steps.append(visible_states.copy())
            
            # Update current samples for next step
            current_samples = visible_states
        
        # Animation update function
        def update(frame):
            step_text.set_text(f"Step: {frame+1}/{n_steps}")
            
            # Get reconstructions for this step
            reconstructions = reconstruction_steps[frame]
            
            # Update plots
            for i in range(n_samples):
                if reshape is not None:
                    # Update image
                    images_recon[i].set_array(reconstructions[i].reshape(reshape))
                else:
                    # Update bars
                    for j, bar in enumerate(images_recon[i]):
                        bar.set_height(reconstructions[i][j])
            
            # Flatten list of images for blit
            all_artists = []
            for imgs in images_orig + images_recon:
                if isinstance(imgs, list):
                    all_artists.extend(imgs)
                else:
                    all_artists.append(imgs)
            
            all_artists.append(step_text)
            return all_artists
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_steps,
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_reconstruction_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Reconstruction animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani