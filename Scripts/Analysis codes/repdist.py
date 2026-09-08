import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt

# --- Helper Functions (Unchanged) ---

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
    and the other moves.
    """
    bottom_pole_z = top_pole_z - instant_length
    if instant_length > 1e-9:
        normalized_value = (raw_coord - bottom_pole_z) / instant_length
        normalized_value = 1 - normalized_value
        return normalized_value
    else:
        return 0.5

# --- NEW: Data Collection Function ---

def collect_replisome_data(percentage, filename_pattern):
    """
    Collects normalized replisome positions for a given percentage of the simulation.
    
    Args:
        percentage (int): The percentage of the simulation time to analyze (e.g., 30).
        filename_pattern (str): A string pattern for the GSD filenames.
        
    Returns:
        list: A list of normalized replisome positions.
    """
    print(f"--- Collecting data for {percentage}% ---")
    
    # Calculate the frame window to analyze
    frame = percentage / 100.0 * 2000
    minframe = int(frame - 20)
    maxframe = int(frame + 20)
    
    # This list will hold data only for this function call
    replisome_data = []

    numbers_to_skip = {}

    for i in range(120):
        if i in numbers_to_skip:
            continue
        
        try:
            input_file = gsd.hoomd.open(filename_pattern.format(i=i), mode='r')
        except FileNotFoundError:
            print(f"Warning: File for run {i} not found. Skipping.")
            continue

        frame_counter = minframe
        for frame in input_file[minframe:maxframe]:
            length = calculate_instantaneous_length(frame_counter, 28.0, 2000)
            typeid_indices = np.where(frame.particles.typeid == 4)[0]
            
            for particle_id in typeid_indices:
                normalized_pos = normalize_coordinate_in_moving_system(
                    frame.particles.position[particle_id][2], length, 28.0
                )
                replisome_data.append(normalized_pos)
            
            frame_counter += 1
    
    print(f"Finished collecting data for {percentage}%. Found {len(replisome_data)} data points.")
    return replisome_data

# --- Main Plotting Script ---

if __name__ == "__main__":
    # Define the percentages you want to plot
    percentages_to_plot = [30, 90]
    colors = ['blue', 'red']  # Colors for each histogram
    
    # Define the pattern for your filenames
    filename_pattern = '1000monruns/100ParA/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru/run_{i}_20k_3.50_fullringrepsoft.gsd'

    # Create a figure and axes for the plot
    plt.figure(figsize=(10, 5))
    
    # Collect data and plot a histogram for each percentage
    for i, p in enumerate(percentages_to_plot):
        # 1. Collect the data
        replisome_positions = collect_replisome_data(p, filename_pattern)
        
        # 2. Plot the histogram for this dataset
        if replisome_positions: # Only plot if data was found
            plt.hist(replisome_positions, 
                     bins=40, 
                     histtype='step', 
                     density=True, 
                     linewidth=2,
                     color=colors[i], 
                     label=f' {p}% of replication time')
    
    # --- Style the final plot ---
    plt.xlim((0, 1))
    plt.xlabel("Normalized Position along Long Axis", fontsize=22)
    plt.ylabel("Probability Density", fontsize=22)
    plt.title("Replisome Distribution at Different Times", fontsize=22)
    plt.legend(fontsize=18) # Display the labels
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig('replisomesidt.png',dpi=300)
    plt.show()