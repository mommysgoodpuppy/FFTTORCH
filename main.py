import torch
import torch.fft
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class FFTMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FFTMLP, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Define frequency bands (4 bands)
        self.num_bands = 4
        
        # First layer combines raw input with frequency band features
        self.first_layer = torch.nn.Linear(input_dim + self.num_bands, hidden_dim)
        
        # Middle layers
        self.middle_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim + self.num_bands, hidden_dim)
            for _ in range(num_layers - 2)
        ])
        
        # Final layer
        self.final_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def get_freq_features(self, x):
        # Compute FFT magnitudes
        x_fft = torch.fft.rfft(x, dim=-1)
        magnitudes = torch.abs(x_fft)
        
        # Split frequencies into bands and compute average magnitude per band
        num_freqs = magnitudes.shape[-1]
        band_size = num_freqs // self.num_bands
        band_features = []
        
        for i in range(self.num_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < self.num_bands - 1 else num_freqs
            band_avg = magnitudes[:, start_idx:end_idx].mean(dim=-1)
            band_features.append(band_avg)
        
        return torch.stack(band_features, dim=-1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        
        # Get initial frequency features
        freq_features = self.get_freq_features(x)
        
        # Combine input with frequency features
        x = torch.cat([x, freq_features], dim=-1)
        
        # First layer
        x = self.first_layer(x)
        x = F.relu(x)  # ReLU is faster than softplus
        
        # Middle layers with frequency feature refresh
        for layer in self.middle_layers:
            # Get fresh frequency features
            freq_features = self.get_freq_features(x)
            x = torch.cat([x, freq_features], dim=-1)
            x = layer(x)
            x = F.relu(x)
        
        # Final layer
        x = self.final_layer(x)
        return x

def train_model(model, train_loader, test_loader, epochs=10, device="cpu"):
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Enable torch optimizations
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch}: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), '
              f'Time: {epoch_time:.2f}s')

def main():
    # MNIST parameters
    input_dim = 28 * 28  # MNIST image size
    hidden_dim = 512
    output_dim = 10      # 10 digits
    num_layers = 3
    batch_size = 256     # Increased batch size further
    epochs = 2

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Use more workers for data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           num_workers=4, pin_memory=True)

    # Create and train model
    model = FFTMLP(input_dim, hidden_dim, output_dim, num_layers)
    train_model(model, train_loader, test_loader, epochs)

if __name__ == "__main__":
    main()
