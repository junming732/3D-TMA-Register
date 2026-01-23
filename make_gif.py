import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import WORKSPACE
import os

def create_jitter_gif(input_file="robust_stack.npy"):
    # 1. Load the stack
    path = os.path.join(WORKSPACE, input_file)
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"Loading {path}...")
    stack = np.load(path)
    print(f"Shape loaded: {stack.shape}")
    
    # 2. Handle Dimensions
    # If shape is (20, 8, Y, X), we pick Channel 0 (DAPI)
    if stack.ndim == 4:
        print("Selecting Channel 0 (DAPI) from 8 channels...")
        stack = stack[:, 0, :, :]
    
    # 3. Setup the Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # Normalize contrast for the first slice so it looks good
    vmin, vmax = np.percentile(stack, (1, 99))
    
    # Draw the first frame
    im = ax.imshow(stack[0], cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    title = ax.set_title("Slice 1")
    
    # 4. Define the Update Function (The loop)
    def update(frame):
        im.set_data(stack[frame])
        title.set_text(f"Slice {frame+1}/20")
        return [im, title]

    # 5. Build and Save
    print("Generating animation (this may take a moment)...")
    ani = animation.FuncAnimation(fig, update, frames=len(stack), interval=200, blit=True)
    
    #save_path = os.path.join(WORKSPACE, "registration_jitter.gif")
    save_path = os.path.join(WORKSPACE, "registration_template-matching.gif")
    ani.save(save_path, writer='pillow', fps=5)
    print(f"DONE! Saved GIF to: {save_path}")

if __name__ == "__main__":
    create_jitter_gif()