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
import hashlib
import argparse
import matplotlib.gridspec as gridspec

def compute_fft_chunk(args):
    start_idx, end_idx, images = args
    ffts = []
    for img in images:
        # Handle color images (3, H, W)
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected color image with shape (3, H, W), got shape {img.shape}")
            
        # Compute 2D FFT for each channel
        fft = torch.fft.rfft2(img, dim=(1, 2))  # Shape: (3, H, W//2 + 1)
        
        # Get magnitude and phase
        magnitude = torch.abs(fft)  # Shape: (3, H, W//2 + 1)
        phase = torch.angle(fft)    # Shape: (3, H, W//2 + 1)
        
        # Flatten the spatial dimensions but keep channels separate
        magnitude = magnitude.reshape(3, -1)  # Shape: (3, H * (W//2 + 1))
        phase = phase.reshape(3, -1)         # Shape: (3, H * (W//2 + 1))
        
        # Stack magnitude and phase, preserving channel information
        fft_combined = torch.stack([magnitude, phase], dim=0)  # Shape: (2, 3, H * (W//2 + 1))
        ffts.append(fft_combined.numpy())
        
    return start_idx, np.array(ffts, dtype=np.float32)

class PrecomputedFFTDataset(Dataset):
    def __init__(self, dataset, cache_dir='fft_cache_selective'):
        self.dataset = dataset
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a unique cache file name based on dataset properties
        dataset_name = type(dataset).__name__
        transform_str = str(dataset.transform) if hasattr(dataset, 'transform') else ''
        cache_hash = hashlib.md5((dataset_name + transform_str).encode()).hexdigest()
        self.cache_file = os.path.join(cache_dir, f'fft_cache_{cache_hash}.npy')
        self.labels_file = os.path.join(cache_dir, f'labels_cache_{cache_hash}.npy')
        
        if os.path.exists(self.cache_file) and os.path.exists(self.labels_file):
            print("Loading cached FFT data...")
            self.fft_cache = np.load(self.cache_file, mmap_mode='r')
            self.labels = np.load(self.labels_file)
        else:
            print("Computing FFT for dataset...")
            # Collect all images and labels
            images = []
            labels = []
            for img, label in tqdm(dataset, desc="Loading images"):
                # Debug: Print first image stats
                if len(images) == 0:
                    print(f"First image in dataset - shape: {img.shape}")
                    print(f"First image min/max: {img.min():.3f}/{img.max():.3f}")
                images.append(img)
                labels.append(label)
            self.labels = np.array(labels)
            
            # Compute FFT in parallel
            chunk_size = 100
            chunks = [(i, min(i + chunk_size, len(images)), images[i:i + chunk_size])
                     for i in range(0, len(images), chunk_size)]
            
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(compute_fft_chunk, chunks),
                    total=len(chunks),
                    desc="Computing FFT"
                ))
            
            # Sort results by index and concatenate
            results.sort(key=lambda x: x[0])
            self.fft_cache = np.concatenate([r[1] for r in results], axis=0)
            
            # Save to cache
            np.save(self.cache_file, self.fft_cache)
            np.save(self.labels_file, self.labels)
            print(f"Cached FFT data to {self.cache_file}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.fft_cache[idx]), self.labels[idx]

class SelectivePhaseNet(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # FFT dimensions
        self.freq_h = input_size
        self.freq_w = input_size//2 + 1
        self.channels = 3
        
        # Calculate feature dimensions
        feature_dim = self.channels * self.freq_h * self.freq_w * 2
        
        # Smaller number of filters per class for efficiency
        self.mag_filters = torch.nn.Parameter(torch.ones(num_classes//2, self.channels, self.freq_h, self.freq_w))
        self.phase_mask = torch.nn.Parameter(torch.ones(num_classes//2, self.channels, self.freq_h, self.freq_w) * 0.5)
        self.phase_temp = torch.nn.Parameter(torch.tensor(1.0))
        self.phase_shifts = torch.nn.Parameter(torch.zeros(num_classes//2, self.channels))
        
        # Efficient MLP with batch norm and residual connections
        hidden_dim = 256  # Reduced hidden dimension
        self.input_bn = nn.BatchNorm1d(feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Better initialization
        nn.init.kaiming_normal_(self.mag_filters)
        nn.init.kaiming_normal_(self.phase_mask)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract and reshape
        magnitude, phase = x[:, 0], x[:, 1]
        magnitude = magnitude.view(batch_size, self.channels, self.freq_h, self.freq_w)
        phase = phase.view(batch_size, self.channels, self.freq_h, self.freq_w)
        
        # Apply filters more efficiently
        mag_out = magnitude.unsqueeze(1) * self.mag_filters.unsqueeze(0)
        
        # Phase processing
        phase_importance = torch.sigmoid(self.phase_mask / self.phase_temp)
        phase_shifts_expanded = self.phase_shifts.view(1, -1, self.channels, 1, 1)
        phase_out = phase.unsqueeze(1) + phase_shifts_expanded
        phase_out = phase_out * phase_importance
        
        # Complex domain operations
        real = mag_out * torch.cos(phase_out)
        imag = mag_out * torch.sin(phase_out)
        
        # Efficient feature processing
        real_avg = real.mean(dim=1)
        imag_avg = imag.mean(dim=1)
        
        features = torch.cat([
            real_avg.reshape(batch_size, -1),
            imag_avg.reshape(batch_size, -1)
        ], dim=1)
        
        # Apply batch norm and MLP
        features = self.input_bn(features)
        return self.mlp(features)

def train(model, train_loader, test_loader, epochs=10, device='cpu', save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Improved optimizer settings
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-4, 'weight_decay': 0.01}
    ])
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device).long()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # L2 regularization is handled by AdamW
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if i % 50 == 49:  # More frequent updates
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f} acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device).long()
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch + 1} Validation Accuracy: {accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning rate: {current_lr:.6f}')
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved with accuracy: {best_acc:.2f}%')

def visualize_filters(model, class_idx, test_loader, save_path=None, precomputed_avg=None):
    from matplotlib.widgets import Slider, Button, CheckButtons
    import math
    
    # CIFAR10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = classes[class_idx]
    
    # Calculate dimensions for visualization
    freq_dim = model.freq_h * model.freq_w * model.channels
    print(f"Model freq_dim: {freq_dim}")
    
    # Get a single example image for the class
    example_idx = None
    class_indices = [i for i, (img, label) in enumerate(test_loader.dataset.dataset) if label == class_idx]
    example_idx = class_indices[10]  # First image of the class
    example_img = test_loader.dataset.dataset[example_idx][0]
    
    # Print debug info about example image selection
    print(f"\nExample image selection:")
    print(f"Selected index: {example_idx}")
    print(f"Class indices available: {len(class_indices)}")
    print(f"Example image shape: {example_img.shape}")
    print(f"Example image stats - min: {example_img.min():.3f}, max: {example_img.max():.3f}")
    
    # Get a few more examples to verify they're different
    if len(class_indices) > 20:
        print("\nChecking multiple examples:")
        for test_idx in class_indices[0:3]:
            test_img = test_loader.dataset.dataset[test_idx][0]
            print(f"Image at index {test_idx} - min: {test_img.min():.3f}, max: {test_img.max():.3f}")
    
    # Denormalize the image using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    orig_img = example_img * std + mean
    print(f"After denorm min/max: {orig_img.min():.3f}/{orig_img.max():.3f}")
    orig_img = torch.clamp(orig_img, 0, 1)
    print(f"After clamp min/max: {orig_img.min():.3f}/{orig_img.max():.3f}")
    
    # Get FFT of original image
    fft = torch.fft.rfft2(orig_img)
    magnitude = torch.abs(fft)
    phase = torch.angle(fft)
    
    print(f"FFT shape: {magnitude.shape}")
    print(f"First test image FFT shape: {test_loader.dataset.fft_cache[0][0].shape}")
    
    # Use actual FFT dimensions for visualization - handle all channels
    c, h, w = magnitude.shape
    
    # Get model's filters and ensure they match FFT dimensions
    filter_idx = class_idx % (model.num_classes // 2)  # Cycle through available filters
    print(f"\nUsing filter index {filter_idx} for class {class_idx}")
    mag_filter = model.mag_filters[filter_idx].detach().cpu()
    phase_mask = torch.sigmoid(model.phase_mask[filter_idx] * torch.exp(model.phase_temp)).detach().cpu()
    learned_phase_shifts = model.phase_shifts[filter_idx].detach().cpu()  # [3] tensor for RGB
    
    # Create full-size filters - match the FFT dimensions
    mag_filter_2d = mag_filter.clone()  # [3, H, W]
    phase_mask_2d = phase_mask.clone()  # [3, H, W]
    
    # Initialize filter responses with default values
    scaled_mag_filter = mag_filter_2d.clone()
    scaled_phase_mask = phase_mask_2d.clone()
    
    # Apply filters with combined phase shift - handle all channels
    total_phase_shifts = learned_phase_shifts.unsqueeze(-1).unsqueeze(-1) + 0.0
    
    # Apply filters in frequency domain
    mag_only_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * phase)
    
    # Keep phase mask strength moderate to preserve detail
    phase_mask_strength = 2.0  # Keep at 2.0 for high-res effect
    phase_shift_strength = np.pi  # Full phase shift range
    
    # Apply phase mask and shifts with controlled effect
    modified_phase = phase * scaled_phase_mask * phase_mask_strength
    # Normalize modified phase to maintain stability
    modified_phase = torch.atan2(torch.sin(modified_phase), torch.cos(modified_phase))  # Wrap to [-pi, pi]
    phase_shifts = total_phase_shifts * phase_shift_strength
    
    phase_only_fft = magnitude * torch.exp(1j * (modified_phase + phase_shifts))
    combined_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * (modified_phase + phase_shifts))
    
    # Convert back to spatial domain with original dimensions
    mag_filtered_img = torch.fft.irfft2(mag_only_fft, s=(h, h*2))
    phase_filtered_img = torch.fft.irfft2(phase_only_fft, s=(h, h*2))
    combined_img = torch.fft.irfft2(combined_fft, s=(h, h*2))
    
    # Normalize the filtered images
    def normalize_image(img):
        img = img - img.min()
        img = img / (img.max() - img.min() + 1e-8)
        return torch.clamp(img, 0, 1)
    
    mag_filtered_img = normalize_image(mag_filtered_img)
    phase_filtered_img = normalize_image(phase_filtered_img)
    combined_img = normalize_image(combined_img)
    
    print("\nFiltered image stats:")
    print(f"Magnitude filtered - min: {mag_filtered_img.min():.3f}, max: {mag_filtered_img.max():.3f}")
    print(f"Phase filtered - min: {phase_filtered_img.min():.3f}, max: {phase_filtered_img.max():.3f}")
    print(f"Combined filtered - min: {combined_img.min():.3f}, max: {combined_img.max():.3f}")
    
    # Convert filter responses to spatial domain for visualization
    mag_filter_spatial = torch.zeros_like(mag_filtered_img)
    phase_mask_spatial = torch.zeros_like(phase_filtered_img)
    for c in range(3):
        # Create frequency domain tensors
        mag_freq = torch.zeros_like(magnitude[c], dtype=torch.complex64)
        phase_freq = torch.zeros_like(magnitude[c], dtype=torch.complex64)
        
        # Set real parts to filter responses
        mag_freq.real = scaled_mag_filter[c]
        phase_freq.real = scaled_phase_mask[c]
        
        # Convert to spatial domain
        mag_filter_spatial[c] = torch.fft.irfft2(mag_freq, s=(h, h*2))
        phase_mask_spatial[c] = torch.fft.irfft2(phase_freq, s=(h, h*2))
    
    # Normalize spatial domain filter responses
    mag_filter_spatial = (mag_filter_spatial - mag_filter_spatial.min()) / (mag_filter_spatial.max() - mag_filter_spatial.min() + 1e-8)
    phase_mask_spatial = (phase_mask_spatial - phase_mask_spatial.min()) / (phase_mask_spatial.max() - phase_mask_spatial.min() + 1e-8)
    
    def create_colored_output(img, filter_response):
        # Ensure correct dimensions [C, H, W]
        if img.dim() == 4:
            img = img.squeeze(0)
        
        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Get local frequency content using gradients
        local_freq = torch.zeros_like(img)
        for i in range(1, img.shape[1]-1):
            for j in range(1, img.shape[2]-1):
                dx = img[:,i+1,j] - img[:,i-1,j]
                dy = img[:,i,j+1] - img[:,i,j-1]
                local_freq[:,i,j] = torch.sqrt(dx*dx + dy*dy)
        
        # Normalize frequency
        local_freq = (local_freq - local_freq.min()) / (local_freq.max() - local_freq.min() + 1e-8)
        
        # Create visualization tensor with correct dimensions
        rgb = torch.zeros(img.shape[1], img.shape[2], 3)
        
        # Fill in RGB channels
        for c in range(3):
            rgb[..., 0] += img[c] / 3  # Original intensity (averaged)
            rgb[..., 1] += img[c] * filter_response[c] / 3  # Filter response (averaged)
            rgb[..., 2] += img[c] * local_freq[c] / 3  # Local frequency (averaged)
        
        # Normalize and ensure [0, 1] range
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = torch.clamp(rgb, 0, 1)
        return rgb
    
    # Create colored outputs - handle dimensions properly
    phase_filtered_rgb = create_colored_output(phase_filtered_img, phase_mask_spatial)
    mag_filtered_rgb = create_colored_output(mag_filtered_img, mag_filter_spatial)
    combined_filtered_rgb = create_colored_output(combined_img, mag_filter_spatial * phase_mask_spatial)
    
    # Create the main figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1.5])
    
    plt.suptitle(f'Interactive Analysis of {class_name}', fontsize=16)
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title('Example Image')
    im1 = ax1.imshow(orig_img.permute(1, 2, 0).numpy(), aspect='equal')
    ax1.axis('off')
    
    # Average original
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title('Average Original')
    if precomputed_avg is not None:
        avg_img = precomputed_avg
    else:
        # Compute average image from this class
        print("\nComputing average image...")
        avg_img = torch.zeros_like(orig_img)
        count = 0
        num_to_average = 1 # Number of images to average
        
        # Get all indices for this class
        class_indices = [i for i, (img, label) in enumerate(test_loader.dataset.dataset) if label == class_idx]
        print(f"Found {len(class_indices)} images for class {class_name}")
        
        # Take a random sample if we have more than num_to_average images
        if len(class_indices) > num_to_average:
            import random
            random.shuffle(class_indices)
            class_indices = class_indices[:num_to_average]
        
        # Compute average
        for idx in class_indices:
            img = test_loader.dataset.dataset[idx][0]
            img = img * std + mean  # Denormalize
            img = torch.clamp(img, 0, 1)
            avg_img += img
            count += 1
            
        if count > 0:
            avg_img /= count
            print(f"Averaged {count} images")
            
            # Enhance the average image
            def enhance_image(img, contrast=25.5, brightness=1.0, color_enhance=1.0):
                # Enhance each channel separately to preserve colors better
                enhanced = torch.zeros_like(img)
                for c in range(3):
                    # Center around 0.5
                    centered = img[c] - 0.5
                    # Apply strong contrast
                    contrasted = centered * contrast
                    # Recenter and apply brightness
                    enhanced_channel = (contrasted + 0.5) * brightness
                    # Normalize channel to use full range
                    enhanced_channel = enhanced_channel - enhanced_channel.min()
                    enhanced_channel = enhanced_channel / (enhanced_channel.max() + 1e-8)
                    # Enhance color saturation
                    enhanced_channel = torch.pow(enhanced_channel, 1/color_enhance)
                    enhanced[c] = enhanced_channel
                
                # Final normalization and clamping
                enhanced = torch.clamp(enhanced, 0, 1)
                return enhanced
            
            # Enhance the average image to make it more vibrant
            avg_img = enhance_image(avg_img)
            print("Enhanced average image with stronger contrast and color preservation")
        else:
            print("Warning: No images found for averaging")
            avg_img = orig_img  # Fallback to example image if no other images found
    
    im2 = ax2.imshow(avg_img.permute(1, 2, 0).numpy(), aspect='equal')
    ax2.axis('off')
    
    # FFT magnitude
    ax3 = plt.subplot(gs[0, 2])
    ax3.set_title('FFT Magnitude')
    
    # Get FFT of average image instead of original for filtering
    fft = torch.fft.rfft2(avg_img)  # Use average image for FFT
    magnitude = torch.abs(fft)
    phase = torch.angle(fft)
    
    freq_vis = torch.log(torch.mean(magnitude, dim=0) + 1)  # Average across channels
    freq_vis = (freq_vis - freq_vis.min()) / (freq_vis.max() - freq_vis.min())  # Normalize
    im3 = ax3.imshow(freq_vis.numpy(), aspect='equal', cmap='magma')
    ax3.axis('off')
    
    # Phase filtered
    ax4 = plt.subplot(gs[1, 0])
    ax4.set_title('Phase Filtered')
    im4 = ax4.imshow(np.zeros((h, h, 3)), aspect='equal', vmin=0, vmax=1)
    ax4.axis('off')
    
    # Magnitude filtered
    ax5 = plt.subplot(gs[1, 1])
    ax5.set_title('Magnitude Filtered')
    im5 = ax5.imshow(np.zeros((h, h, 3)), aspect='equal', vmin=0, vmax=1)
    ax5.axis('off')
    
    # Combined filtered
    ax6 = plt.subplot(gs[1, 2])
    ax6.set_title('Combined Filtered')
    im6 = ax6.imshow(np.zeros((h, h, 3)), aspect='equal', vmin=0, vmax=1)
    ax6.axis('off')
    
    # Frequency response
    ax6f = plt.subplot(gs[1, 3])
    ax6f.set_title('Frequency Response')
    im6f = ax6f.imshow(np.zeros((h, h, 3)), aspect='equal', vmin=0, vmax=1)
    ax6f.axis('off')
    
    # Accuracy and frequency analysis
    ax7 = plt.subplot(gs[2, :2])
    ax7.set_title('Model Confidence for Each Class')
    bar_container = ax7.bar(range(10), torch.zeros(10))
    ax7.set_xticks(range(10))
    ax7.set_xticklabels(classes, rotation=45, ha='right')
    ax7.set_ylim(0, 1)
    ax7.set_ylabel('Confidence')
    
    # Add frequency spectrum analysis
    ax8 = plt.subplot(gs[2, 2:4])
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
        temp = s_temp.val
        use_log = c_mag_log.get_status()[0]
        
        # Update filters - handle all channels
        if use_log:
            scaled_mag_filter = torch.exp(mag_scale * torch.log(mag_filter_2d + 1) + mag_shift) - 1
        else:
            scaled_mag_filter = mag_filter_2d * mag_scale + mag_shift
            
        scaled_phase_mask = torch.sigmoid(phase_mask_2d * temp) * phase_scale
        
        # Apply filters with combined phase shift - handle all channels
        total_phase_shifts = learned_phase_shifts.unsqueeze(-1).unsqueeze(-1) + manual_phase_shift
        
        # Apply filters in frequency domain
        mag_only_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * phase)
        
        # Keep phase mask strength moderate to preserve detail
        phase_mask_strength = 2.0  # Keep at 2.0 for high-res effect
        phase_shift_strength = np.pi  # Full phase shift range
        
        # Apply phase mask and shifts with controlled effect
        modified_phase = phase * scaled_phase_mask * phase_mask_strength
        # Normalize modified phase to maintain stability
        modified_phase = torch.atan2(torch.sin(modified_phase), torch.cos(modified_phase))  # Wrap to [-pi, pi]
        phase_shifts = total_phase_shifts * phase_shift_strength
        
        phase_only_fft = magnitude * torch.exp(1j * (modified_phase + phase_shifts))
        combined_fft = (magnitude * scaled_mag_filter) * torch.exp(1j * (modified_phase + phase_shifts))
        
        # Convert back to spatial domain with original dimensions
        mag_filtered_img = torch.fft.irfft2(mag_only_fft, s=(h, h*2))
        phase_filtered_img = torch.fft.irfft2(phase_only_fft, s=(h, h*2))
        combined_img = torch.fft.irfft2(combined_fft, s=(h, h*2))
        
        # Normalize the filtered images
        def normalize_image(img):
            img = img - img.min()
            img = img / (img.max() - img.min() + 1e-8)
            return torch.clamp(img, 0, 1)
        
        mag_filtered_img = normalize_image(mag_filtered_img)
        phase_filtered_img = normalize_image(phase_filtered_img)
        combined_img = normalize_image(combined_img)
        
        # Convert filter responses to spatial domain for visualization
        mag_filter_spatial = torch.zeros_like(mag_filtered_img)
        phase_mask_spatial = torch.zeros_like(phase_filtered_img)
        for c in range(3):
            # Create frequency domain tensors
            mag_freq = torch.zeros_like(magnitude[c], dtype=torch.complex64)
            phase_freq = torch.zeros_like(magnitude[c], dtype=torch.complex64)
            
            # Set real parts to filter responses
            mag_freq.real = scaled_mag_filter[c]
            phase_freq.real = scaled_phase_mask[c]
            
            # Convert to spatial domain
            mag_filter_spatial[c] = torch.fft.irfft2(mag_freq, s=(h, h*2))
            phase_mask_spatial[c] = torch.fft.irfft2(phase_freq, s=(h, h*2))
        
        # Normalize spatial domain filter responses
        mag_filter_spatial = (mag_filter_spatial - mag_filter_spatial.min()) / (mag_filter_spatial.max() - mag_filter_spatial.min() + 1e-8)
        phase_mask_spatial = (phase_mask_spatial - phase_mask_spatial.min()) / (phase_mask_spatial.max() - phase_mask_spatial.min() + 1e-8)
        
        def create_colored_output(img, filter_response):
            # Ensure correct dimensions [C, H, W]
            if img.dim() == 4:
                img = img.squeeze(0)
            
            # Normalize image to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Get local frequency content using gradients
            local_freq = torch.zeros_like(img)
            for i in range(1, img.shape[1]-1):
                for j in range(1, img.shape[2]-1):
                    dx = img[:,i+1,j] - img[:,i-1,j]
                    dy = img[:,i,j+1] - img[:,i,j-1]
                    local_freq[:,i,j] = torch.sqrt(dx*dx + dy*dy)
            
            # Normalize frequency
            local_freq = (local_freq - local_freq.min()) / (local_freq.max() - local_freq.min() + 1e-8)
            
            # Create visualization tensor with correct dimensions
            rgb = torch.zeros(img.shape[1], img.shape[2], 3)
            
            # Fill in RGB channels
            for c in range(3):
                rgb[..., 0] += img[c] / 3  # Original intensity (averaged)
                rgb[..., 1] += img[c] * filter_response[c] / 3  # Filter response (averaged)
                rgb[..., 2] += img[c] * local_freq[c] / 3  # Local frequency (averaged)
            
            # Normalize and ensure [0, 1] range
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = torch.clamp(rgb, 0, 1)
            return rgb
        
        # Create colored outputs - handle dimensions properly
        phase_filtered_rgb = create_colored_output(phase_filtered_img, phase_mask_spatial)
        mag_filtered_rgb = create_colored_output(mag_filtered_img, mag_filter_spatial)
        combined_filtered_rgb = create_colored_output(combined_img, mag_filter_spatial * phase_mask_spatial)
        
        # Update image displays with correct dimensions
        im4.set_data(phase_filtered_rgb)
        im5.set_data(mag_filtered_rgb)
        im6.set_data(combined_filtered_rgb)
        
        # Update frequency analysis - average across channels and reshape for plotting
        freq_x = torch.arange(magnitude.shape[-1])  # Frequency bins
        orig_spectrum = torch.mean(torch.abs(magnitude), dim=(0,1))  # Average across channels and batch
        filtered_spectrum = torch.mean(torch.abs(magnitude * scaled_mag_filter), dim=(0,1))
        
        # Update line plots with matching dimensions
        spectrum_line.set_data(freq_x, orig_spectrum.detach().cpu().numpy())
        filtered_line.set_data(freq_x, filtered_spectrum.detach().cpu().numpy())
        
        ax8.relim()
        ax8.autoscale_view()
        
        # Calculate frequency statistics - average across channels
        total_energy = torch.sum(magnitude * magnitude)
        filtered_energy = torch.sum(magnitude * magnitude * scaled_mag_filter * scaled_mag_filter)
        energy_ratio = filtered_energy / total_energy
        
        # Calculate frequency band distribution - average across channels
        freq_bands = torch.tensor_split(torch.mean(magnitude, dim=(0,1)), 4)
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
            f'Learned (R,G,B): {np.degrees(learned_phase_shifts[0]):.1f}°, '
            f'{np.degrees(learned_phase_shifts[1]):.1f}°, '
            f'{np.degrees(learned_phase_shifts[2]):.1f}°\n'
            f'Manual: {np.degrees(manual_phase_shift):.1f}°\n'
            f'Total (R,G,B): {np.degrees(learned_phase_shifts[0] + manual_phase_shift):.1f}°, '
            f'{np.degrees(learned_phase_shifts[1] + manual_phase_shift):.1f}°, '
            f'{np.degrees(learned_phase_shifts[2] + manual_phase_shift):.1f}°'
        )
        
        # Update model confidence
        with torch.no_grad():
            # Move model to CPU for visualization
            model.cpu()
            
            # Apply current filters to test examples
            test_outputs = []
            for img_np in test_loader.dataset.fft_cache[:10]:  # Get 10 examples
                # Convert numpy memmap to torch tensor
                img = torch.from_numpy(img_np.copy())  # Need .copy() for memmap
                
                # Get magnitude and phase components
                img_mag = img[0]  # Already flattened
                img_phase = img[1]  # Already flattened
                
                # Reshape to match model's expected dimensions
                img_mag = img_mag.reshape(3, h, -1)  # [3, 32, 17]
                img_phase = img_phase.reshape(3, h, -1)  # [3, 32, 17]
                
                # Create the modified input tensor
                modified_mag = img_mag.clone()
                modified_phase = img_phase.clone()
                
                # Apply current filter settings
                if use_log:
                    modified_mag = torch.exp(mag_scale * torch.log(modified_mag + 1) + mag_shift) - 1
                else:
                    modified_mag = modified_mag * mag_scale + mag_shift
                
                # Apply phase modifications
                modified_phase = modified_phase * phase_scale + total_phase_shifts
                
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
            h, w = orig_img.shape[1:]
            response_2d = torch.zeros_like(orig_img)
            
            # Map the 1D filter response back to frequency space
            freq_dim = h * w // 2 + 1
            response_1d = filter_response.flatten()[:freq_dim]
            
            # Create radial mapping
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            freq_dist_radial = torch.sqrt((x - w//2)**2 + (y - h//2)**2)
            freq_dist_radial = (freq_dist_radial / freq_dist_radial.max() * (freq_dim-1)).long()
            
            # Map to 2D
            for i in range(h):
                for j in range(w):
                    idx = min(freq_dist_radial[i,j].item(), freq_dim-1)
                    response_2d[:,i,j] = response_1d[idx]
            
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
        im4.set_data(phase_filtered_norm.permute(1, 2, 0))
        im5.set_data(mag_filtered_norm.permute(1, 2, 0))
        im6.set_data(combined_norm.permute(1, 2, 0))
        
        # Update only combined frequency response
        combined_response_vis = get_frequency_response(scaled_phase_mask * scaled_mag_filter)
        im6f.set_data(combined_response_vis.permute(1, 2, 0))
        
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
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train and visualize SelectivePhaseNet on CIFAR10')
    parser.add_argument('--viz', action='store_true', help='Skip training and only run visualization')
    parser.add_argument('--model-path', type=str, default='models_cifar/best_model.pth', help='Path to model for visualization')
    parser.add_argument('--digit', type=int, default=None, help='Specific class to visualize (0-9)')
    args = parser.parse_args()

    # For 32x32 images
    input_size = 32  # Image size
    output_dim = 10
    batch_size = 256
    epochs = 20
    num_workers = 4
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Keep color information
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Only create training dataset if we're not just visualizing
    if not args.viz:
        train_dataset = PrecomputedFFTDataset(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform),
            cache_dir='fft_cache_selective_cifar'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True, persistent_workers=True)
    
    # We need test dataset for both training and visualization
    test_dataset = PrecomputedFFTDataset(
        datasets.CIFAR10('../data', train=False, transform=transform),
        cache_dir='fft_cache_selective_cifar'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True)
    
    # Create model
    model = SelectivePhaseNet(input_size, output_dim)
    
    if args.viz:
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        # Get CIFAR10 class names
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
        
        if args.digit is not None:
            print(f"Visualizing class: {classes[args.digit]}")
            num_avg = 200
            class_indices = [i for i, (img, label) in enumerate(test_loader.dataset.dataset) if label == args.digit]
            avg_img = torch.zeros(3, 32, 32)  # RGB image
            count = 0
            
            # First get the average original image and test examples
            with torch.no_grad():
                for data, labels in test_loader:
                    mask = labels == args.digit
                    if mask.any():
                        # Get original images from the dataset
                        orig_dataset = test_loader.dataset.dataset
                        # Only get images of the current class
                        orig_imgs = []
                        for i in range(len(orig_dataset)):
                            img, label = orig_dataset[i]
                            if label == args.digit:
                                if len(orig_imgs) >= num_avg:  # Use num_avg instead of max_images
                                    break
                                # Keep color channels
                                if isinstance(img, torch.Tensor):
                                    img = img.squeeze()
                                else:
                                    img = torch.tensor(img).squeeze()
                                orig_imgs.append(img)
                        
                        if orig_imgs:
                            orig_imgs = torch.stack(orig_imgs)
                            # Properly normalize the average
                            avg_img = orig_imgs.mean(0)
                            # Denormalize using ImageNet stats
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            avg_img = avg_img * std + mean
                            # Ensure in [0, 1] range
                            avg_img = torch.clamp(avg_img, 0, 1)
                            count = len(orig_imgs)
                            print(f"Averaged {count} images for class {classes[args.digit]}")
                            break
            
            if count == 0:
                print(f"No images found for class {classes[args.digit]}")
                return
            
            # Create visualization with the computed average
            visualize_filters(model, args.digit, test_loader, 
                            f'visualizations_cifar/class_{classes[args.digit]}_analysis.png',
                            precomputed_avg=avg_img)
        else:
            print("Visualizing all classes...")
            os.makedirs('visualizations_cifar', exist_ok=True)
            for digit in range(10):
                print(f"Visualizing class: {classes[digit]}")
                visualize_filters(model, digit, test_loader, 
                                f'visualizations_cifar/class_{classes[digit]}_analysis.png')
    else:
        train(model, train_loader, test_loader, epochs, save_dir='models_cifar')
    
    # Visualize filters for each class
    os.makedirs('visualizations_cifar', exist_ok=True)
    for digit in range(10):
        visualize_filters(model, digit, test_loader, f'visualizations_cifar/class_{digit}_analysis.png')

if __name__ == "__main__":
    main()
