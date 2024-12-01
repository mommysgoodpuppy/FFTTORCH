import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import os
from main_selective_phase import SelectivePhaseNet, PrecomputedFFTDataset

def create_3d_phase_visualization(model, digit, test_loader, num_phase_steps=50):
    # Get average image and FFT for this digit
    avg_img = torch.zeros(28, 28)

    # First get the average original image
    with torch.no_grad():
        for data, labels in test_loader:
            mask = labels == digit
            if mask.any():
                # Get original images from dataset before FFT
                orig_dataset = test_loader.dataset.dataset
                orig_imgs = [orig_dataset[i][0] for i in range(len(orig_dataset)) if orig_dataset[i][1] == digit]
                orig_imgs = torch.stack(orig_imgs)
                avg_img = orig_imgs.mean(0).squeeze()
                break

    # Get FFT of average image
    fft = torch.fft.rfft2(avg_img)
    magnitude = torch.abs(fft)
    phase = torch.angle(fft)

    # Create the figure with both 2D and 3D plots
    fig = plt.figure(figsize=(15, 8))

    # 2D visualization (left subplot)
    ax_2d = fig.add_subplot(121)

    # Create phase scale slider with range [0, 2]
    slider_ax = plt.axes([0.15, 0.02, 0.3, 0.03])
    phase_slider = Slider(slider_ax, 'Phase Scale', 0.0, 2.0, valinit=0.0)

    # Create bias slider
    bias_ax = plt.axes([0.15, 0.06, 0.3, 0.03])
    bias_slider = Slider(bias_ax, 'Visibility Threshold', 0.0, 1.0, valinit=0.5)

    # Create power slider with extended range
    power_ax = plt.axes([0.15, 0.10, 0.3, 0.03])
    power_slider = Slider(power_ax, 'Compression Power', 0.1, 10.0, valinit=1.2)

    # Calculate reconstructed images at different phase scales
    responses = []
    z = np.linspace(0, 2.0, num_phase_steps)  # Changed to [0, 2]

    for phase_scale in z:
        # Apply phase scaling in frequency domain
        scaled_phase = phase.numpy() * phase_scale
        fft_complex = magnitude.numpy() * np.exp(1j * scaled_phase)

        # Convert back to spatial domain
        img = torch.fft.irfft2(torch.from_numpy(fft_complex))

        # Normalize and enhance contrast
        img_flat = img.flatten()
        low = torch.quantile(img_flat, 0.02)  # 2nd percentile
        high = torch.quantile(img_flat, 0.98)  # 98th percentile
        img = torch.clamp((img - low) / (high - low + 1e-8), 0, 1)

        responses.append(img)

    volume = torch.stack(responses, dim=-1)

    # Initial 2D plot
    im = ax_2d.imshow(responses[0], cmap='magma')
    plt.colorbar(im, ax=ax_2d)
    ax_2d.set_title('Reconstructed Image')

    # 3D visualization (right subplot)
    ax_3d = fig.add_subplot(122, projection='3d')

    # Create meshgrid for 3D visualization
    x = np.arange(volume.shape[0])
    y = np.arange(volume.shape[1])
    z = np.arange(volume.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Normalize volume to [0, 1]
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min())

    def get_compressed_points(vol_norm, power, threshold):
        # Apply non-linear transformation to values
        vol_compressed = vol_norm ** power

        # Get points above threshold
        visible_points = vol_compressed > threshold
        if not visible_points.any():  # Return None if no points are visible
            return None, None, None, None

        x_points = X[visible_points]
        y_points = Z[visible_points]  # Phase dimension
        z_points = Y[visible_points]

        # Move points based on their values, but only in phase dimension (y_points)
        values = vol_norm[visible_points].numpy()  # Convert to numpy

        # Find the center of the phase dimension
        y_mean = y_points.mean()
        # Compress only in phase dimension
        y_points = y_mean + (y_points - y_mean) * values.reshape(-1)  # Ensure shapes match

        intensities = vol_compressed[visible_points].numpy()  # Convert to numpy
        return z_points, y_points, x_points, intensities  # Rotate to correct orientation

    # Initial scatter plot
    x_points, y_points, z_points, intensities = get_compressed_points(volume_norm, power_slider.val, bias_slider.val)
    scatter = ax_3d.scatter(x_points, y_points, z_points,
                           c=intensities, cmap='plasma',  # plasma gives good depth perception while maintaining contrast
                           alpha=intensities, s=8)

    # Create a semi-transparent plane for the current phase scale
    xx, yy = np.meshgrid(np.arange(28), np.arange(28))
    initial_z = np.zeros_like(xx)

    ax_3d.set_title('3D Phase Space')
    ax_3d.set_facecolor('black')
    ax_3d.grid(False)

    # Set axis labels
    ax_3d.set_xlabel('X')
    ax_3d.set_zlabel('Y')  # Swapped Y and Z
    ax_3d.set_ylabel('Phase Scale')

    # Remove axis panes and set background to black
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('none')
    ax_3d.yaxis.pane.set_edgecolor('none')
    ax_3d.zaxis.pane.set_edgecolor('none')
    ax_3d.set_facecolor((0, 0, 0, 0))
    fig.patch.set_facecolor('white')  # Keep overall figure background white

    # Set better viewing angle
    ax_3d.view_init(elev=20, azim=-60)

    # Create the initial plane
    plane_surface = []
    # Add transparent plane
    plane_surface.append(ax_3d.plot_surface(xx, initial_z, yy, alpha=0.1, color='white'))
    # Add border lines
    plane_surface.append(ax_3d.plot([0, 27], [0, 0], [0, 0], color='cyan', linewidth=2)[0])  # bottom edge
    plane_surface.append(ax_3d.plot([0, 27], [0, 0], [27, 27], color='cyan', linewidth=2)[0])  # top edge
    plane_surface.append(ax_3d.plot([0, 0], [0, 0], [0, 27], color='cyan', linewidth=2)[0])  # left edge
    plane_surface.append(ax_3d.plot([27, 27], [0, 0], [0, 27], color='cyan', linewidth=2)[0])  # right edge

    scatter_points = [scatter]  # Store scatter plot in a list for updating

    def update(val):
        # Update 2D visualization
        phase_scale = phase_slider.val
        idx = int((phase_scale / 2.0) * (num_phase_steps-1))
        im.set_array(responses[idx])

        # Update the plane position
        for p in plane_surface:
            if hasattr(p, 'remove'):
                p.remove()
        new_z = np.full_like(xx, idx)
        plane_surface[0] = ax_3d.plot_surface(xx, new_z, yy, alpha=0.1, color='white')
        # Update border lines at new position
        plane_surface[1] = ax_3d.plot([0, 27], [new_z[0,0], new_z[0,0]], [0, 0], color='cyan', linewidth=2)[0]  # bottom edge
        plane_surface[2] = ax_3d.plot([0, 27], [new_z[0,0], new_z[0,0]], [27, 27], color='cyan', linewidth=2)[0]  # top edge
        plane_surface[3] = ax_3d.plot([0, 0], [new_z[0,0], new_z[0,0]], [0, 27], color='cyan', linewidth=2)[0]  # left edge
        plane_surface[4] = ax_3d.plot([27, 27], [new_z[0,0], new_z[0,0]], [0, 27], color='cyan', linewidth=2)[0]  # right edge

        # Update scatter plot with new threshold and power
        x_points, y_points, z_points, intensities = get_compressed_points(
            volume_norm, power_slider.val, bias_slider.val)

        # Remove old scatter plot
        if hasattr(scatter_points[0], 'remove'):
            scatter_points[0].remove()

        # Create new scatter plot only if we have points to show
        if x_points is not None:
            scatter_points[0] = ax_3d.scatter(
                x_points,
                y_points,
                z_points,
                c=intensities,
                cmap="inferno",
                alpha=intensities,
                s=21,
            )
        else:
            scatter_points[0] = None  # Clear reference when no points

        fig.canvas.draw_idle()

    phase_slider.on_changed(update)
    bias_slider.on_changed(update)
    power_slider.on_changed(update)
    update(0)

    plt.show()

def main():
    # Load the model and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset with FFT precomputation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    base_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_dataset = PrecomputedFFTDataset(base_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    # Initialize model architecture
    input_dim = 784  # 28x28 MNIST images
    model = SelectivePhaseNet(input_dim // 2 + 1, num_classes=10)
    
    # Load the trained model
    model_path = os.path.join('models', 'selective_phase_model_best.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print(f"No saved model found at {model_path}. Please train the model first using main_selective_phase.py")
        return
    
    model.eval()
    
    # Create visualization for digit 0
    create_3d_phase_visualization(model, digit=8, test_loader=test_loader)

if __name__ == "__main__":
    main()
