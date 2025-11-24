import re
import matplotlib.pyplot as plt

# Read the slurm output
with open('slurm-16339.out', 'r') as f:
    lines = f.readlines()

train_epochs = []
train_losses = []
val_epochs = []
val_losses = []
val_psnr = []
val_ssim = []

for line in lines:
    # Training loss
    train_match = re.search(r'epoch:\s+(\d+),.*?loss:\s+([\d.]+)', line)
    if train_match and '[train]' in line:
        train_epochs.append(int(train_match.group(1)))
        train_losses.append(float(train_match.group(2)))
    
    # Validation metrics
    val_match = re.search(r'loss:\s+([\d.]+),\s+psnr:\s+([\d.]+),\s+ssim:\s+([\d.]+)', line)
    if val_match and '[valid]' in line:
        val_losses.append(float(val_match.group(1)))
        val_psnr.append(float(val_match.group(2)))
        val_ssim.append(float(val_match.group(3)))

# Get unique validation epochs (every 5 epochs)
val_epochs = list(range(5, 75, 5))[:len(val_losses)]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Train vs Val Loss
axes[0, 0].plot(train_epochs, train_losses, 'b-', alpha=0.3, linewidth=0.5, label='Train')
axes[0, 0].plot(val_epochs, val_losses, 'ro-', linewidth=2, markersize=6, label='Validation')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Validation PSNR
axes[0, 1].plot(val_epochs, val_psnr, 'g^-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].set_title('Validation PSNR')
axes[0, 1].grid(True, alpha=0.3)

# Validation SSIM
axes[1, 0].plot(val_epochs, val_ssim, 'ms-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].set_title('Validation SSIM')
axes[1, 0].grid(True, alpha=0.3)

# All metrics normalized
axes[1, 1].plot(val_epochs, val_psnr, 'g^-', label='PSNR/10', linewidth=2)
axes[1, 1].plot(val_epochs, val_ssim, 'ms-', label='SSIM', linewidth=2)
axes[1, 1].plot(val_epochs, [l*100 for l in val_losses], 'ro-', label='Loss×100', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Metric Value')
axes[1, 1].set_title('All Validation Metrics')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_val_metrics.png', dpi=300, bbox_inches='tight')
print(f"Training samples: {len(train_losses)}")
print(f"Validation points: {len(val_losses)}")
print(f"Best Val Loss: {min(val_losses):.6f} at epoch {val_epochs[val_losses.index(min(val_losses))]}")
print(f"Best Val PSNR: {max(val_psnr):.2f} at epoch {val_epochs[val_psnr.index(max(val_psnr))]}")
