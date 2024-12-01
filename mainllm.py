import torch
import torch.fft
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import requests
import zipfile
import io

def download_wikitext2(root='wikitext-2'):
    """Download WikiText-2 dataset if not already downloaded."""
    if os.path.exists(root):
        print("WikiText-2 already downloaded")
        return
    
    print("Downloading WikiText-2...")
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall('.')
    print("Download complete!")

class TextPreprocessor:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.freqs = {}
    
    def build_vocab(self, texts):
        # Count frequencies
        for text in texts:
            for word in text.split():
                self.freqs[word] = self.freqs.get(word, 0) + 1
        
        # Add frequent words to vocabulary
        idx = len(self.vocab)
        for word, freq in self.freqs.items():
            if freq >= self.min_freq and word not in self.vocab:
                self.vocab[word] = idx
                idx += 1
    
    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]

class WikiText2Dataset(Dataset):
    def __init__(self, file_path, preprocessor, max_seq_len):
        self.max_seq_len = max_seq_len
        self.preprocessor = preprocessor
        
        # Load and preprocess text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode full text
        tokens = self.preprocessor.encode(text)
        
        # Create sequences with overlap
        self.sequences = []
        for i in range(0, len(tokens) - max_seq_len, max_seq_len // 2):
            seq = tokens[i:i + max_seq_len]
            if len(seq) == max_seq_len:
                self.sequences.append(torch.tensor(seq))
        
        self.sequences = torch.stack(self.sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def load_wikitext2(split, root='wikitext-2'):
    """Load and preprocess WikiText-2 dataset."""
    # Download if needed
    download_wikitext2(root)
    
    # Create preprocessor if training split
    if split == 'train':
        preprocessor = TextPreprocessor(min_freq=2)
        with open(os.path.join(root, 'wikitext-2-v1', f'wiki.{split}.tokens'), 'r', encoding='utf-8') as f:
            train_text = f.read()
        preprocessor.build_vocab([train_text])
        print(f"Vocabulary size: {len(preprocessor.vocab)}")
    
    # Load appropriate split
    file_path = os.path.join(root, 'wikitext-2-v1', f'wiki.{split}.tokens')
    dataset = WikiText2Dataset(file_path, preprocessor, max_seq_len=128)
    return dataset.sequences

class PrecomputedFFTDataset(Dataset):
    def __init__(self, sequences, cache_dir='fft_cache'):
        self.sequences = sequences
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f'fft_cache_{len(sequences)}.pt')
        sequences_file = os.path.join(cache_dir, f'sequences_cache_{len(sequences)}.pt')
        
        if os.path.exists(cache_file) and os.path.exists(sequences_file):
            print("Loading pre-computed FFTs from cache...")
            # Load directly into shared memory
            self.fft_cache = torch.load(cache_file)
            self.sequences = torch.load(sequences_file)
            # Move to shared memory for efficient access
            self.fft_cache.share_memory_()
            self.sequences.share_memory_()
        else:
            print("Pre-computing FFTs using multiple processes...")
            # Convert sequences to numpy for multiprocessing
            sequences_np = sequences.numpy()
            
            # Prepare data in chunks for multiprocessing
            chunk_size = 1000
            chunks = []
            
            for i in range(0, len(sequences), chunk_size):
                end = min(i + chunk_size, len(sequences))
                chunks.append((i, end, sequences_np[i:end]))
            
            # Process chunks in parallel
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(compute_sequence_fft_chunk, chunks),
                    total=len(chunks),
                    desc="Computing FFTs"
                ))
            
            # Combine results directly into PyTorch tensors
            results.sort(key=lambda x: x[0])
            all_ffts = np.concatenate([r[1] for r in results])
            self.fft_cache = torch.from_numpy(all_ffts)
            self.sequences = sequences
            
            # Move to shared memory
            self.fft_cache.share_memory_()
            self.sequences.share_memory_()
            
            # Save as PyTorch tensors
            torch.save(self.fft_cache, cache_file)
            torch.save(self.sequences, sequences_file)
            print("FFT pre-computation complete and cached!")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # No copy needed as we're using shared memory
        return self.fft_cache[idx], self.sequences[idx]

class FFTLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len=128, hidden_dim=256):
        super(FFTLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.freq_dim = max_seq_len // 2 + 1
        self.hidden_dim = hidden_dim
        
        # Initialize frequency domain embeddings directly
        self.token_freq_embeddings = torch.nn.Parameter(
            torch.randn(vocab_size, hidden_dim, self.freq_dim, dtype=torch.cfloat) * 0.02
        )
        
        # Position embeddings in frequency domain
        pos_freqs = torch.zeros(max_seq_len, hidden_dim, self.freq_dim, dtype=torch.cfloat)
        for i in range(max_seq_len):
            phase = 2 * math.pi * i / max_seq_len
            t = torch.arange(self.freq_dim, dtype=torch.float32)
            pos_freqs[i] = torch.exp(1j * phase * t).unsqueeze(0).expand(hidden_dim, -1) * 0.02
        
        self.pos_freq_embeddings = torch.nn.Parameter(pos_freqs)
        
        # Output projection
        self.output_proj = torch.nn.Linear(hidden_dim, vocab_size)
        
        # Layer norm in frequency domain (on magnitudes)
        self.freq_layer_norm = torch.nn.LayerNorm([hidden_dim, self.freq_dim])
    
    def forward(self, x):
        # x is already in frequency domain from preprocessing
        # Shape: [batch_size, freq_dim]
        batch_size = x.shape[0]
        
        # Convert to complex
        x = torch.complex(x, torch.zeros_like(x))
        
        # Get token frequency embeddings
        # Shape: [batch_size, hidden_dim, freq_dim]
        token_freqs = self.token_freq_embeddings[x]
        
        # Add position frequency embeddings
        # Shape: [hidden_dim, freq_dim]
        pos_freqs = self.pos_freq_embeddings[0]  # Use first position for now
        
        # Combine in frequency domain
        # Shape: [batch_size, hidden_dim, freq_dim]
        combined_freqs = token_freqs + pos_freqs
        
        # Layer norm on magnitudes in frequency domain
        magnitudes = torch.abs(combined_freqs)
        phases = torch.angle(combined_freqs)
        normed_magnitudes = self.freq_layer_norm(magnitudes)
        combined_freqs = normed_magnitudes * torch.exp(1j * phases)
        
        # Sum frequency components to get final features
        # Shape: [batch_size, hidden_dim]
        features = combined_freqs.real.sum(dim=-1)
        
        # Project to vocabulary size
        # Shape: [batch_size, vocab_size]
        logits = self.output_proj(features)
        
        return logits

def compute_sequence_fft_chunk(args):
    start_idx, end_idx, sequences = args
    print(f"Processing chunk {start_idx} to {end_idx}")
    ffts = []
    for seq in sequences:
        # Convert sequence to frequency domain
        seq_float = seq.astype(np.float32)
        fft = np.fft.rfft(seq_float)
        ffts.append(fft)
    return start_idx, np.array(ffts, dtype=np.complex64)

def train_model(model, train_loader, test_loader, epochs=10, device="cpu"):
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Maximize CPU parallelization
    torch.set_num_threads(torch.get_num_threads())
    torch.set_float32_matmul_precision('medium')
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        
        for batch_idx, (ffts, sequences) in enumerate(train_loader):
            ffts, sequences = ffts.to(device), sequences.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with pre-computed FFTs
            logits = model(ffts)
            
            # Use next token prediction loss
            loss = criterion(logits[:, :-1].reshape(-1, model.vocab_size), 
                           sequences[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(ffts)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')

def main():
    # Load and preprocess WikiText-2 dataset
    train_dataset = load_wikitext2('train')
    test_dataset = load_wikitext2('test')
    
    # Wrap datasets with FFT pre-computation
    train_dataset = PrecomputedFFTDataset(train_dataset)
    test_dataset = PrecomputedFFTDataset(test_dataset)
    
    # Maximize data loading parallelization
    num_workers = min(8, torch.get_num_threads())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=16,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True)
    
    vocab_size = 10000  # From WikiText-2 preprocessing
    model = FFTLanguageModel(vocab_size)
    
    try:
        # Test a batch
        print("\nTesting first batch...")
        ffts, sequences = next(iter(train_loader))
        print(f"FFT data shape: {ffts.shape}")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Sample sequence: {sequences[0][:10]}")
        
        # Forward pass test
        logits = model(ffts)
        print(f"Output logits shape: {logits.shape}")
        
        # Train model
        train_model(model, train_loader, test_loader, epochs=10)
        
        # Test memory usage
        print("\nMemory usage:")
        print(f"Token embeddings: {model.token_freq_embeddings.numel() * 8 / 1024 / 1024:.2f} MB")
        print(f"Position embeddings: {model.pos_freq_embeddings.numel() * 8 / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error in test batch: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
