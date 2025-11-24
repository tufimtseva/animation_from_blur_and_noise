import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the largest event file (most recent/complete training run)
event_files = glob.glob('/w/20251/mahikav/experiments/gopro_mbd_mahika/events.out.tfevents.*')
largest_file = max(event_files, key=lambda x: os.path.getsize(x))
print(f"Reading from: {largest_file}")

train_losses = []
steps = []

for event in summary_iterator(largest_file):
    for value in event.summary.value:
        if value.tag == 'train/loss':
            train_losses.append(value.simple_value)
            steps.append(event.step)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, train_losses, 'b-', linewidth=1, alpha=0.7)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss over Steps (74 Epochs)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('training_loss_tensorboard.png', dpi=150, bbox_inches='tight')
print(f"Total steps: {len(steps)}")
if train_losses:
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Starting loss: {train_losses[0]:.6f}")
print("Saved to training_loss_tensorboard.png")
