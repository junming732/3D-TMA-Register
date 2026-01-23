import tifffile
import numpy as np
import os
from config import TMA_FOLDERS, WORKSPACE

# --- INPUT FROM QUPATH (Level 0 metrics) ---
PIXEL_SIZE_L0 = 0.4961
X_MICRONS = 1668.8
Y_MICRONS = 1557.0
SIZE_MICRONS = 1200

# --- DOWNSAMPLE STRATEGY ---
# Level 0 = Full Res (Too big for network drive, causes Bus Error)
# Level 2 = 4x smaller (Perfect for visualization)
DOWNSAMPLE = 4 

# Calculate coordinates for Level 2
# divide pixels by 4 to match the smaller image
X_L0_PX = (X_MICRONS - (SIZE_MICRONS / 2)) / PIXEL_SIZE_L0
Y_L0_PX = (Y_MICRONS - (SIZE_MICRONS / 2)) / PIXEL_SIZE_L0
BOX_L0_PX = SIZE_MICRONS / PIXEL_SIZE_L0

X_START = int(X_L0_PX / DOWNSAMPLE)
Y_START = int(Y_L0_PX / DOWNSAMPLE)
BOX_SIZE = int(BOX_L0_PX / DOWNSAMPLE)
X_END = X_START + BOX_SIZE
Y_END = Y_START + BOX_SIZE

def harvest_level2():
    print(f"Targeting Level 2 (4x smaller). Box: {X_START}:{X_END}, {Y_START}:{Y_END}")
    
    stack = []
    
    for i, folder_path in enumerate(TMA_FOLDERS):
        # Build path
        folder_name = os.path.basename(folder_path)
        file_name = f"{folder_name}_Scan1.unmixed.qptiff"
        full_path = os.path.join(folder_path, "Scan1", file_name)
        
        try:
            with tifffile.TiffFile(full_path) as tif:
                # Access Level 2 (Index 2 in the pyramid)
                page = tif.series[0].levels[2]
                
                # Load the WHOLE level into RAM (Safe for 1.7GB)
                # We explicitly use 'asarray()' to copy to RAM, avoiding the Bus Error
                full_level = page.asarray()
                
                # Check dimensions and Cut
                if full_level.ndim == 3: # (8, H, W)
                    crop = full_level[:, Y_START:Y_END, X_START:X_END]
                else:
                    crop = full_level[Y_START:Y_END, X_START:X_END]

                stack.append(crop)
                print(f"Slice {i+1}/20: Success")
                
        except IndexError:
            print(f"Slice {i+1}/20: FAILED - Level 2 not found in file.")
        except MemoryError:
             print(f"Slice {i+1}/20: FAILED - Server RAM full.")
        except Exception as e:
            print(f"Slice {i+1}/20: Error: {e}")

    if len(stack) > 0:
        output_path = os.path.join(WORKSPACE, "single_patient_stack.npy")
        np.save(output_path, np.array(stack))
        print(f"\nSUCCESS: Saved med-res stack to {output_path}")

if __name__ == "__main__":
    harvest_level2()