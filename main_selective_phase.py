import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

def compute_fft_chunk(args):
    start_idx, end_idx, images = args
    ffts = []
    for img in images:
        fft = torch.fft.rfft(img.view(-1))
        # Get magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Store both full magnitude and phase
        # Let the model decide dynamically which phases to use
        fft_combined = torch.stack([magnitude, phase], dim=0).numpy()
        ffts.append(fft_combined)
    return start_idx, np.array(ffts, dtype=np.float32)

class PrecomputedFFTDataset(Dataset):
    def __init__(self, dataset, cache_dir='fft_cache_selective'):
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f'fft_cache_{len(dataset)}.pt')
        labels_file = os.path.join(cache_dir, f'labels_cache_{len(dataset)}.pt')
        
        if os.path.exists(cache_file) and os.path.exists(labels_file):
            print("Loading pre-computed FFTs from cache...")
            self.fft_cache = torch.load(cache_file)
            self.labels = torch.load(labels_file)
            self.fft_cache.share_memory_()
            self.labels.share_memory_()
        else:
            print("Pre-computing FFTs using multiple processes...")
            chunk_size = 1000
            chunks = []
            all_images = []
            all_labels = []
            
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                all_images.append(img)
                all_labels.append(label)
            
            for i in range(0, len(dataset), chunk_size):
                end = min(i + chunk_size, len(dataset))
                chunks.append((i, end, all_images[i:end]))
            
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(compute_fft_chunk, chunks),
                    total=len(chunks),
                    desc="Computing FFTs"
                ))
            
            results.sort(key=lambda x: x[0])
            all_ffts = np.concatenate([r[1] for r in results])
            self.fft_cache = torch.from_numpy(all_ffts)
            self.labels = torch.tensor([label for label in all_labels], dtype=torch.long)
            
            self.fft_cache.share_memory_()
            self.labels.share_memory_()
            
            torch.save(self.fft_cache, cache_file)
            torch.save(self.labels, labels_file)
            print("FFT pre-computation complete and cached!")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.fft_cache[idx], self.labels[idx]

class SelectivePhaseNet(nn.Module):
    def __init__(self, freq_dim, num_classes=10):
        super().__init__()
        self.freq_dim = freq_dim
        self.num_classes = num_classes
        
        # Learnable magnitude filters (static per class)
        self.mag_filters = torch.nn.Parameter(torch.ones(num_classes, freq_dim))
        
        # Learnable phase importance (static mask per class)
        self.phase_mask = torch.nn.Parameter(torch.ones(num_classes, freq_dim) * 0.5)
        self.phase_temp = torch.nn.Parameter(torch.tensor(1.0))
        
        # Learnable phase shifts (dynamic shift per class)
        self.phase_shifts = torch.nn.Parameter(torch.zeros(num_classes))
        
        # Initialize with small random values
        nn.init.normal_(self.mag_filters, mean=1.0, std=0.02)
        nn.init.normal_(self.phase_mask, mean=0.0, std=0.02)
        nn.init.normal_(self.phase_shifts, mean=0.0, std=0.5)  # Larger std for phase shifts
    
    def forward(self, x):
        # Split into magnitude and phase
        magnitude = x[:, 0]  # [batch, freq_dim]
        phase = x[:, 1]     # [batch, freq_dim]
        
        phase_masks = torch.sigmoid(self.phase_mask * torch.exp(self.phase_temp))
        
        # Compute both contributions in parallel
        mag_out = magnitude.unsqueeze(1) * self.mag_filters
        
        # Apply phase shifts before computing contribution
        shifted_phase = phase.unsqueeze(1) + self.phase_shifts.unsqueeze(-1)  # [batch, num_classes, freq_dim]
        phase_contribution = torch.sin(shifted_phase) * phase_masks.unsqueeze(0)
        
        # Sum both contributions
        energies = mag_out.sum(dim=-1) + phase_contribution.sum(dim=-1)
        
        # Track phase usage for monitoring
        with torch.no_grad():
            self.last_phase_usage = (phase_masks > 0.5).float().mean(dim=1)
        
        return energies

def train(model, train_loader, test_loader, epochs=10, device='cpu'):
    optimizer = torch.optim.Adam([
        {'params': [model.mag_filters], 'lr': 0.001},
        {'params': [model.phase_mask], 'lr': 0.001},
        {'params': [model.phase_shifts], 'lr': 0.01},  # Higher learning rate for phase shifts
        {'params': [model.phase_temp], 'lr': 0.001}
    ])
    
    print(f"Training on {device}")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    torch.set_num_threads(torch.get_num_threads())
    torch.set_float32_matmul_precision('medium')
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            
            # Add small regularization to prevent phase usage from growing too high
            phase_usage = torch.sigmoid(model.phase_mask).mean()
            loss = loss + 0.01 * phase_usage
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        model.eval()
        correct = 0
        phase_usage_per_class = torch.zeros(model.num_classes)
        samples_per_class = torch.zeros(model.num_classes)
        phase_shifts = model.phase_shifts.detach().cpu().numpy()
        phase_shifts_deg = np.degrees(phase_shifts)
        print(f'Learned phase shifts: ' + ' '.join([f'{i}: {v:.1f}°' for i, v in enumerate(phase_shifts_deg)]))
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                # Track phase usage per class
                magnitude = data[:, 0]
                phase_masks = torch.sigmoid(model.phase_mask * torch.exp(model.phase_temp))
                for i in range(model.num_classes):
                    mask = target == i
                    if mask.any():
                        phase_usage_per_class[i] += phase_masks[i].sum()
                        samples_per_class[i] += mask.sum() * model.freq_dim

        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Calculate average phase usage per class
        phase_usage_percent = 100. * phase_usage_per_class / (samples_per_class + 1e-8)
        print(f'Epoch {epoch}: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        print(f'Phase usage per class: ' + ' '.join([f'{i}: {v:.1f}%' for i, v in enumerate(phase_usage_percent)]))
        print(f'Temperature: {torch.exp(model.phase_temp).item():.2f}, Time: {time.time() - start_time:.2f}s')

def visualize_filters(model, digit, test_loader, save_path=None):
    from matplotlib.widgets import Slider, Button, CheckButtons
    import math
    
    # Calculate dimensions for visualization
    freq_dim = model.freq_dim
    print(f"Model freq_dim: {freq_dim}")
    
    # Get average image and FFT for this digit from test set
    avg_img = torch.zeros(28, 28)
    
    # First get the average original image and test examples
    with torch.no_grad():
        for data, labels in test_loader:
            mask = labels == digit
            if mask.any():
                # Get original images from dataset before FFT
                orig_dataset = test_loader.dataset.dataset
                orig_imgs = [orig_dataset[i][0] for i in range(len(orig_dataset)) if orig_dataset[i][1] == digit]
                orig_imgs = torch.stack(orig_imgs)
                avg_img = orig_imgs.mean(0).squeeze()
                
                # Get FFT test examples (already in FFT format)
                test_imgs = data[mask][:10]  # Get 10 examples
                test_labels = labels[mask][:10]
                break
    
    # Get FFT of average image
    fft = torch.fft.rfft2(avg_img)
    magnitude = torch.abs(fft)
    phase = torch.angle(fft)
    
    print(f"FFT shape: {magnitude.shape}")
    print(f"First test image FFT shape: {test_imgs[0][0].shape}")
    
    # Use actual FFT dimensions for visualization
    h, w = magnitude.shape
    
    # Get model's filters
    mag_filter = model.mag_filters[digit].detach().cpu()
    phase_mask = torch.sigmoid(model.phase_mask[digit] * torch.exp(model.phase_temp)).detach().cpu()
    learned_phase_shift = model.phase_shifts[digit].item()  # Store learned shift
    
    # Create full-size filters
    mag_filter_2d = torch.zeros_like(magnitude)
    phase_mask_2d = torch.zeros_like(magnitude)
    mag_filter_2d.flatten()[:freq_dim] = mag_filter
    phase_mask_2d.flatten()[:freq_dim] = phase_mask
    
    # Initialize filter responses with default values
    scaled_mag_filter = mag_filter_2d.clone()
    scaled_phase_mask = phase_mask_2d.clone()
    
    # Create the figure
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 4)  # Reduced to 4 columns
    
    plt.suptitle(f'Interactive Analysis of Digit {digit}', fontsize=16)
    
    # Original image and its spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(avg_img, cmap='magma')
    ax1.set_title('Average Original')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1:3])
    freq_vis = torch.log(magnitude + 1)
    im2 = ax2.imshow(freq_vis, cmap='magma')
    ax2.set_title('Frequency Spectrum')
    plt.colorbar(im2, ax=ax2)
    
    # Add frequency cross-section plot
    ax2_cross = fig.add_subplot(gs[0, 3])
    center_row = freq_vis[freq_vis.shape[0]//2, :]
    center_col = freq_vis[:, 0]
    ax2_cross.plot(center_row, label='Horizontal', alpha=0.7)
    ax2_cross.plot(center_col, label='Vertical', alpha=0.7)
    ax2_cross.set_title('Frequency Cross-section')
    ax2_cross.legend()
    ax2_cross.grid(True)
    
    # Results row with filtered images and combined frequency response
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(torch.zeros_like(avg_img), cmap='magma', vmin=0, vmax=1)
    ax4.set_title(f'After Phase Mask')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(torch.zeros_like(avg_img), cmap='magma', vmin=0, vmax=1)
    ax5.set_title('After Magnitude Filter')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(torch.zeros_like(avg_img), cmap='magma', vmin=0, vmax=1)
    ax6.set_title('Combined Result')
    plt.colorbar(im6, ax=ax6)
    
    ax6f = fig.add_subplot(gs[1, 3])  # Only keeping combined frequency response
    im6f = ax6f.imshow(torch.zeros_like(avg_img), cmap='magma', vmin=0, vmax=1)
    ax6f.set_title('Frequency Response')
    ax6f.axis('off')
    
    # Accuracy and frequency analysis
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.set_title('Model Confidence for Each Digit')
    bar_container = ax7.bar(range(10), torch.zeros(10))
    ax7.set_ylim(0, 1)
    ax7.set_xlabel('Digit')
    ax7.set_ylabel('Confidence')
    
    # Add frequency spectrum analysis
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.set_title('Frequency Response')
    spectrum_line, = ax8.plot([], [], 'b-', alpha=0.7, label='Original')
    filtered_line, = ax8.plot([], [], 'r--', alpha=0.7, label='Filtered')
    ax8.set_yscale('log')
    ax8.grid(True)
    ax8.legend()
    
    # Add frequency energy text and stats
    stats_text = ax7.text(1.02, 0.5, '', transform=ax7.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='center')
    
    # Add sliders and controls
    plt.subplots_adjust(bottom=0.35)
    
    # Magnitude controls
    ax_mag_scale = plt.axes([0.1, 0.25, 0.3, 0.03])
    ax_mag_shift = plt.axes([0.1, 0.20, 0.3, 0.03])
    ax_mag_log = plt.axes([0.1, 0.15, 0.15, 0.03])
    
    # Phase controls
    ax_phase_scale = plt.axes([0.1, 0.10, 0.3, 0.03])
    ax_phase_shift = plt.axes([0.1, 0.05, 0.3, 0.03])
    ax_temp = plt.axes([0.5, 0.25, 0.3, 0.03])
    
    # Reset button
    ax_reset = plt.axes([0.5, 0.05, 0.1, 0.03])
    
    s_mag_scale = Slider(ax_mag_scale, 'Magnitude Scale', 0.0, 5.0, valinit=1.0)
    s_mag_shift = Slider(ax_mag_shift, 'Magnitude Shift', -2.0, 2.0, valinit=0.0)
    s_phase_scale = Slider(ax_phase_scale, 'Phase Scale', 0.0, 2.0, valinit=1.0)
    s_phase_shift = Slider(ax_phase_shift, 'Phase Shift', -math.pi, math.pi, valinit=0.0)
    s_temp = Slider(ax_temp, 'Temperature', 0.1, 10.0, valinit=torch.exp(model.phase_temp).item())
    
    b_reset = Button(ax_reset, 'Reset')
    c_mag_log = CheckButtons(ax_mag_log, ['Log Magnitude'], [False])
    
    def update(val=None):
        # Get slider values
        mag_scale = s_mag_scale.val
        mag_shift = s_mag_shift.val
        phase_scale = s_phase_scale.val
        manual_phase_shift = s_phase_shift.val
        total_phase_shift = manual_phase_shift + learned_phase_shift  # Combine shifts
        temp = s_temp.val
        use_log = c_mag_log.get_status()[0]
        
        # Update filters
        if use_log:
            scaled_mag_filter = torch.exp(mag_scale * torch.log(mag_filter_2d + 1) + mag_shift) - 1
        else:
            scaled_mag_filter = mag_filter_2d * mag_scale + mag_shift
            
        scaled_phase_mask = torch.sigmoid(phase_mask_2d * temp) * phase_scale
        
        # Apply filters with combined phase shift
        mag_only_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * phase)
        phase_only_fft = magnitude * torch.exp(1j * ((phase * scaled_phase_mask) + total_phase_shift))
        combined_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * ((phase * scaled_phase_mask) + total_phase_shift))
        
        # Convert back to spatial domain
        mag_filtered_img = torch.fft.irfft2(mag_only_fft)
        phase_filtered_img = torch.fft.irfft2(phase_only_fft)
        combined_img = torch.fft.irfft2(combined_fft)
        
        # Update images with color
        def create_colored_output(img, filter_response):
            # Normalize image to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Get local frequency content using gradients
            local_freq = torch.zeros_like(img)
            for i in range(1, img.shape[0]-1):
                for j in range(1, img.shape[1]-1):
                    dx = img[i+1,j] - img[i-1,j]
                    dy = img[i,j+1] - img[i,j-1]
                    local_freq[i,j] = torch.sqrt(dx*dx + dy*dy)
            
            # Normalize frequency
            local_freq = (local_freq - local_freq.min()) / (local_freq.max() - local_freq.min() + 1e-8)
            
            # Create two different frequency response mappings
            h, w = img.shape
            response_2d_radial = torch.zeros_like(img)
            response_2d_rect = torch.zeros_like(img)
            
            # Map the 1D filter response back to frequency space
            freq_dim = h * w // 2 + 1  # Size of FFT output for real input
            response_1d = filter_response.flatten()[:freq_dim]
            
            # 1. Radial mapping (creates medallion patterns)
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            freq_dist_radial = torch.sqrt((x - w//2)**2 + (y - h//2)**2)
            freq_dist_radial = (freq_dist_radial / freq_dist_radial.max() * (freq_dim-1)).long()
            
            # 2. Rectangular mapping (follows actual FFT frequency structure)
            freq_y = torch.abs(y - h//2) / (h//2)
            freq_x = x / w  # For real FFT, x frequencies are naturally ordered
            freq_dist_rect = (torch.sqrt(freq_y**2 + freq_x**2) * (freq_dim-1)).long()
            
            # Map both types
            for i in range(h):
                for j in range(w):
                    idx_radial = min(freq_dist_radial[i,j].item(), freq_dim-1)
                    idx_rect = min(freq_dist_rect[i,j].item(), freq_dim-1)
                    response_2d_radial[i,j] = response_1d[idx_radial]
                    response_2d_rect[i,j] = response_1d[idx_rect]
            
            # Blend both responses
            blend_factor = 0.7  # Adjust this to control medallion vs. actual FFT structure
            response_2d = blend_factor * response_2d_radial + (1 - blend_factor) * response_2d_rect
            
            # Normalize the blended response
            response_2d = (response_2d - response_2d.min()) / (response_2d.max() - response_2d.min() + 1e-8)
            
            # Create RGB image combining intensity, frequency response, and local frequency
            rgb = torch.stack([
                img,  # Red: intensity
                img * response_2d,  # Green: blended frequency response
                img * local_freq,  # Blue: local frequency content
            ], dim=-1)
            
            # Ensure all values are in [0, 1]
            rgb = torch.clamp(rgb, 0, 1)
            return rgb
        
        # Get current filter responses
        phase_response = scaled_phase_mask
        mag_response = scaled_mag_filter
        combined_response = phase_response * mag_response
        
        # Create colored outputs
        phase_filtered_rgb = create_colored_output(phase_filtered_img, phase_response)
        mag_filtered_rgb = create_colored_output(mag_filtered_img, mag_response)
        combined_filtered_rgb = create_colored_output(combined_img, combined_response)
        
        # Update image displays
        im4.set_data(phase_filtered_img)
        im5.set_data(mag_filtered_img)
        im6.set_data(combined_img)
        
        # Update frequency analysis
        orig_spectrum = torch.mean(torch.abs(magnitude), dim=0)
        filtered_spectrum = torch.mean(torch.abs(magnitude * scaled_mag_filter), dim=0)
        
        spectrum_line.set_data(range(len(orig_spectrum)), orig_spectrum)
        filtered_line.set_data(range(len(filtered_spectrum)), filtered_spectrum)
        
        ax8.relim()
        ax8.autoscale_view()
        
        # Calculate frequency statistics
        total_energy = torch.sum(magnitude * magnitude)
        filtered_energy = torch.sum(magnitude * magnitude * scaled_mag_filter * scaled_mag_filter)
        energy_ratio = filtered_energy / total_energy
        
        # Calculate frequency band distribution
        freq_bands = torch.tensor_split(magnitude, 4)
        band_energies = [torch.sum(band * band).item() for band in freq_bands]
        total_band_energy = sum(band_energies)
        band_percentages = [e/total_band_energy * 100 for e in band_energies]
        
        stats_text.set_text(
            f'Energy Preserved: {energy_ratio:.1%}\n\n'
            f'Frequency Bands:\n'
            f'Low: {band_percentages[0]:.1f}%\n'
            f'Mid-Low: {band_percentages[1]:.1f}%\n'
            f'Mid-High: {band_percentages[2]:.1f}%\n'
            f'High: {band_percentages[3]:.1f}%\n\n'
            f'Phase Shifts:\n'
            f'Learned: {np.degrees(learned_phase_shift):.1f}°\n'
            f'Manual: {np.degrees(manual_phase_shift):.1f}°\n'
            f'Total: {np.degrees(total_phase_shift):.1f}°'
        )
        
        # Update model confidence
        with torch.no_grad():
            # Move model to CPU for visualization
            model.cpu()
            
            # Apply current filters to test examples
            test_outputs = []
            for img in test_imgs:
                # Get magnitude and phase components
                img_mag = img[0]  # Already flattened
                img_phase = img[1]  # Already flattened
                
                # Create the modified input tensor
                modified_mag = img_mag.clone()
                modified_phase = img_phase.clone()
                
                # Apply current filter settings
                if use_log:
                    modified_mag = torch.exp(mag_scale * torch.log(modified_mag + 1) + mag_shift) - 1
                else:
                    modified_mag = modified_mag * mag_scale + mag_shift
                
                # Apply phase modifications
                modified_phase = modified_phase * phase_scale + total_phase_shift
                
                # Stack into model input format
                modified_input = torch.stack([modified_mag, modified_phase], dim=0).unsqueeze(0)
                
                # Get model prediction
                output = model(modified_input)
                output = torch.nn.functional.softmax(output, dim=1)
                test_outputs.append(output)
            
            # Average confidences across test examples
            avg_confidence = torch.mean(torch.cat(test_outputs, dim=0), dim=0)
            
            # Update bar plot
            for rect, val in zip(bar_container, avg_confidence):
                rect.set_height(val.item())
        
        # Get frequency response visualizations
        def get_frequency_response(filter_response):
            h, w = avg_img.shape
            response_2d = torch.zeros_like(avg_img)
            
            # Map the 1D filter response back to 2D frequency space
            freq_dim = h * w // 2 + 1
            response_1d = filter_response.flatten()[:freq_dim]
            
            # Create radial mapping
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            freq_dist = torch.sqrt((x - w//2)**2 + (y - h//2)**2)
            freq_dist = (freq_dist / freq_dist.max() * (freq_dim-1)).long()
            
            # Map to 2D
            for i in range(h):
                for j in range(w):
                    idx = min(freq_dist[i,j].item(), freq_dim-1)
                    response_2d[i,j] = response_1d[idx]
            
            # Ensure response is properly normalized and detached
            response_2d = response_2d.detach().cpu()
            response_norm = (response_2d - response_2d.min()) / (response_2d.max() - response_2d.min() + 1e-8)
            return response_norm
        
        # Update main images with enhanced contrast
        def enhance_contrast(img, percentile=2):
            img_flat = img.flatten()
            low = torch.quantile(img_flat, percentile/100)
            high = torch.quantile(img_flat, 1 - percentile/100)
            img_norm = torch.clamp((img - low) / (high - low + 1e-8), 0, 1)
            return img_norm
        
        # Normalize and enhance contrast for main images
        phase_filtered_norm = enhance_contrast(phase_filtered_img)
        mag_filtered_norm = enhance_contrast(mag_filtered_img)
        combined_norm = enhance_contrast(combined_img)
        
        # Update displays
        im4.set_data(phase_filtered_norm)
        im5.set_data(mag_filtered_norm)
        im6.set_data(combined_norm)
        
        # Update only combined frequency response
        combined_response_vis = get_frequency_response(scaled_phase_mask * scaled_mag_filter)
        im6f.set_data(combined_response_vis.numpy())
        
        fig.canvas.draw_idle()
    
    def reset(event):
        s_mag_scale.set_val(1.0)
        s_mag_shift.set_val(0.0)
        s_phase_scale.set_val(1.0)
        s_phase_shift.set_val(0.0)
        s_temp.set_val(torch.exp(model.phase_temp).item())
        c_mag_log.set_active(0)
        update()
    
    # Connect controls to update function
    s_mag_scale.on_changed(update)
    s_mag_shift.on_changed(update)
    s_phase_scale.on_changed(update)
    s_phase_shift.on_changed(update)
    s_temp.on_changed(update)
    c_mag_log.on_clicked(update)
    b_reset.on_clicked(reset)
    
    # Initial update
    update()
    
    plt.show()
    plt.close()

def main():
    input_dim = 28 * 28
    output_dim = 10
    batch_size = 256
    epochs = 12
    num_workers = 4
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = PrecomputedFFTDataset(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        cache_dir='fft_cache_selective'
    )
    
    test_dataset = PrecomputedFFTDataset(
        datasets.MNIST('../data', train=False, transform=transform),
        cache_dir='fft_cache_selective'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True)
    
    model = SelectivePhaseNet(input_dim // 2 + 1, output_dim)
    train(model, train_loader, test_loader, epochs)
    
    # Visualize filters for each digit
    os.makedirs('visualizations', exist_ok=True)
    for digit in range(10):
        visualize_filters(model, digit, test_loader, f'visualizations/digit_{digit}_analysis.png')

if __name__ == "__main__":
    main()
