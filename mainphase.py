import torch
import torch.fft
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os
import math

def compute_fft_chunk(args):
    start_idx, end_idx, images = args
    ffts = []
    for img in images:
        fft = torch.fft.rfft(img.view(-1))
        fft_combined = torch.stack([fft.real, fft.imag], dim=0).numpy()
        ffts.append(fft_combined)
    return start_idx, np.array(ffts, dtype=np.float32)

class PrecomputedFFTDataset(Dataset):
    def __init__(self, dataset, cache_dir='fft_cache'):
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

class FFTMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFTMLP, self).__init__()
        self.freq_dim = input_dim // 2 + 1
        self.num_classes = output_dim
        
        filters_real = torch.zeros(output_dim, self.freq_dim)
        filters_imag = torch.zeros(output_dim, self.freq_dim)
        for i in range(output_dim):
            phase = 2 * math.pi * i / output_dim
            t = torch.arange(self.freq_dim, dtype=torch.float32)
            complex_filter = torch.exp(1j * phase * t) * 0.02
            filters_real[i] = complex_filter.real
            filters_imag[i] = complex_filter.imag
        
        self.filters_real = torch.nn.Parameter(filters_real)
        self.filters_imag = torch.nn.Parameter(filters_imag)
    
    def forward(self, x):
        # Split input into real and imaginary parts
        x_real = x[:, 0]  # First channel is real
        x_imag = x[:, 1]  # Second channel is imaginary
        
        # Complex multiplication with broadcasting
        # x shape: [batch, freq_dim]
        # filters shape: [num_classes, freq_dim]
        # Result shape: [batch, num_classes, freq_dim]
        y_real = (x_real.unsqueeze(1) * self.filters_real) - (x_imag.unsqueeze(1) * self.filters_imag)
        y_imag = (x_real.unsqueeze(1) * self.filters_imag) + (x_imag.unsqueeze(1) * self.filters_real)
        
        # Sum along frequency dimension
        energies = y_real.sum(dim=-1) + y_imag.sum(dim=-1)
        
        return energies

def train_model(model, train_loader, test_loader, epochs=10, device="cpu"):
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch {epoch}: Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%), Time: {time.time() - start_time:.2f}s')

def main():
    input_dim = 28 * 28
    hidden_dim = 512  
    output_dim = 10
    batch_size = 256
    epochs = 3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Clear cache directory
    import shutil
    cache_dir = 'fft_cache'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_dataset = PrecomputedFFTDataset(train_dataset)
    test_dataset = PrecomputedFFTDataset(test_dataset)
    
    num_workers = min(8, torch.get_num_threads())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True)

    model = FFTMLP(input_dim, hidden_dim, output_dim)
    train_model(model, train_loader, test_loader, epochs)

if __name__ == "__main__":
    main()
