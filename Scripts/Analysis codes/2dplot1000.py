import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys

def calculate_instantaneous_length(frame_index, initial_L, growth_duration_frames):
    """
    Calculates the total instantaneous length of the system.
    Grows exponentially from initial_L to 2*initial_L.
    """
    if 0 <= frame_index <= growth_duration_frames:
        k = np.log(2.0) / growth_duration_frames
        length = initial_L * np.exp(k * frame_index)
    else:
        length = 2.0 * initial_L
    return length

def normalize_coordinate_in_moving_system(raw_coord, instant_length, top_pole_z):
    """
    Correctly normalizes a coordinate within a system where one pole is fixed
    and the other moves. The range is [bottom_pole, top_pole].
    
    Returns a normalized value from 0 (bottom pole) to 1 (top pole).
    """
    # Calculate the position of the moving bottom pole
    bottom_pole_z = top_pole_z - instant_length
    
    # The total length is the denominator
    # This avoids division by zero if length is ever zero
    if instant_length > 1e-9:
        # General normalization formula: (value - min) / (max - min)
        # Here: (raw_coord - bottom_pole_z) / (top_pole_z - bottom_pole_z)
        normalized_value = (raw_coord - bottom_pole_z) / instant_length
        normalized_value=1-normalized_value
        return normalized_value
    else:
        return 0.5 # Default value if length is zero

def plot_colored_trajectory(
    gsd_file,
    particle_id,
    initial_L,
    growth_duration_frames,
    top_pole_z,
    long_axis_dim='z',
    short_axis_dim='y',
    frame_skip_rate=1
):
    """
    Plots a particle trajectory with a physically correct time-dependent normalization.
    """
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except FileNotFoundError:
        print(f"Error: GSD file '{gsd_file}' not found.")
        sys.exit(1)

    if frame_skip_rate < 1: frame_skip_rate = 1

    normalized_long_coords = []
    processed_short_coords = []
    processed_timesteps = []
    
    dim_map = {'x': 0, 'y': 1, 'z': 2}
    long_idx = dim_map[long_axis_dim]
    short_idx = dim_map[short_axis_dim]

    # --- 1. Read Data and Process Frame by Frame with Correct Normalization ---
    for frame_index, frame in enumerate(traj[:2005]):
        if frame_index % frame_skip_rate == 0:
            if particle_id < frame.particles.N:
                pos = frame.particles.position[particle_id]
                raw_long_coord = pos[long_idx]
                
                # Get the instantaneous total length for THIS specific frame
                instant_length = calculate_instantaneous_length(frame_index, initial_L, growth_duration_frames)
                
                # Use the new, correct normalization function
                normalized_long = normalize_coordinate_in_moving_system(raw_long_coord, instant_length, top_pole_z)
                
                # Store the processed values
                normalized_long_coords.append(normalized_long)
                processed_short_coords.append(pos[short_idx])
                processed_timesteps.append(frame.configuration.step*0.01/100000)
            
    if not processed_timesteps:
        print(f"Error: No data found for particle {particle_id}.")
        return

    x_coords = np.array(normalized_long_coords)
    y_coords = np.array(processed_short_coords)
    timesteps = np.array(processed_timesteps)

    # --- 2. Create the Plot ---
    fig, ax = plt.subplots(figsize=(16,7))
    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(timesteps.min(), timesteps.max()))
    lc.set_array(timesteps[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # --- 3. Style the Plot ---
    ax.autoscale_view()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel('Normalized long-axis coordinate', fontsize=28)
    ax.set_ylabel('Short-axis coordinate (a)', fontsize=28)
    ax.tick_params(
                    axis='both',
                    which='both',
                    direction='in',
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    labelsize=30,
                    width=2,
                    length=15
                    )

    # The moving pole (bottom) is at 0.0, the fixed pole (top) is at 1.0
    ax.text(0, -0.14, 'Old pole', transform=ax.transAxes, ha='center', fontsize=25,color='r')
    ax.text(1, -0.14, 'New pole', transform=ax.transAxes, ha='center', fontsize=25,color='r')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='both', which='major', labelsize=22)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.12, pad=0.02)

    cbar.set_label(r'$\tau\ (\tau_{\mathrm{rep}})$',
                fontsize=35, fontweight='bold')

    cbar.ax.tick_params(labelsize=28, width=2, length=15)
# plt.xlabel(r'$\tau\ (\tau_{\mathrm{replication}})$', fontsize=25, fontweight='bold')
    # --- 4. Add Top Annotations ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim((-3.5,3.5))
    ax2.tick_params(
                    axis='both',
                    which='both',
                    direction='in',
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    labelsize=30,
                    width=2,
                    length=15
                    )
    # ax2.tick_params(axis='both', which='major', labelsize=22)
    # slow_region = [0.3, 0.6]
    # fast_region = [0.6, 0.9]
    # anchored_region = [0.9, 1.0] 
    # ax2.set_xticks([np.mean(slow_region), np.mean(fast_region), np.mean(anchored_region)])
    # ax2.set_xticklabels(['slow', 'fast', 'anchored'])
    # ax2.tick_params(axis='x', length=0, pad=8)
    # label_colors = ['blue', 'red', 'green']
    # for ticklabel, color in zip(ax2.get_xticklabels(), label_colors):
    #     ticklabel.set_color(color)
    # ax.annotate('', xy=(slow_region[0], 1.08), xytext=(slow_region[1], 1.08), xycoords='axes fraction', arrowprops=dict(arrowstyle='<->', color='blue'))
    # ax.annotate('', xy=(fast_region[0], 1.08), xytext=(fast_region[1], 1.08), xycoords='axes fraction', arrowprops=dict(arrowstyle='<->', color='red'))
    # ax.annotate('', xy=(anchored_region[0], 1.08), xytext=(anchored_region[1], 1.08), xycoords='axes fraction', arrowprops=dict(arrowstyle='<->', color='green'))

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("2D_trajectory_1000.pdf", dpi=400)
    plt.show()

# =============================================================================
# --- Main execution block ---
# =============================================================================
if __name__ == '__main__':
    
    # === USER SETTINGS: EDIT THESE VALUES FOR YOUR SIMULATION ===
    
    GSD_FILENAME = "1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru1/run_204_20k_3.50_fullringrepsoft.gsd"
    PARTICLE_TO_TRACK = 1000
    
    LONG_AXIS = 'z'
    SHORT_AXIS = 'y'
    
    # --- Parameters for your dynamic normalization logic ---
    
    # The initial total length of the system (from z=0 to z=25).
    INITIAL_LENGTH_L = 28.0
    
    # The fixed Z-coordinate of the top ("Old") pole.
    TOP_POLE_Z = 28.0
    
    # The last frame number of the growth phase.
    GROWTH_DURATION_FRAMES = 2000
    
    # Set how many frames to skip.
    FRAME_SKIP =10
    
    # =========================================================================

    plot_colored_trajectory(
        gsd_file=GSD_FILENAME,
        particle_id=PARTICLE_TO_TRACK,
        initial_L=INITIAL_LENGTH_L,
        growth_duration_frames=GROWTH_DURATION_FRAMES,
        top_pole_z=TOP_POLE_Z,
        long_axis_dim=LONG_AXIS,
        short_axis_dim=SHORT_AXIS,
        frame_skip_rate=FRAME_SKIP
    )