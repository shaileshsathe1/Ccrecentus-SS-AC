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
        # normalized_value = 28 - normalized_value
        return normalized_value
    else:
        return 0

# --- NEW: Data Collection Function ---

def collect_replisome_data_per_run(percentage, filename_pattern):
    """
    Collect normalized replisome positions separately for each trajectory.
    
    Returns:
        dict: {run_id: [normalized positions]}
    """
    print(f"--- Collecting data for {percentage}% ---")

    frame = percentage / 100.0 * 2000
    minframe = int(frame - 20)
    maxframe = int(frame + 20)

    numbers_to_skip = {}

    replisome_data = {}

    for i in range(30):
        if i in numbers_to_skip:
            continue

        try:
            input_file = gsd.hoomd.open(filename_pattern.format(i=i), mode='r')
        except FileNotFoundError:
            print(f"Warning: File for run {i} not found. Skipping.")
            continue

        traj_positions = []
        frame_counter = minframe

        for frame in input_file[minframe:maxframe]:
            length = calculate_instantaneous_length(frame_counter, 28.0, 2000)
            typeid_indices = np.where(frame.particles.typeid == 4)[0]

            for pid in typeid_indices:
                normalized_pos = normalize_coordinate_in_moving_system(
                    frame.particles.position[pid][2], length, 28.0
                )
                traj_positions.append(normalized_pos)

            frame_counter += 1

        if traj_positions:
            replisome_data[i] = traj_positions

    print(f"Finished collecting data for {percentage}%. "
          f"Found {len(replisome_data)} trajectories.")
    return replisome_data



def collect_replisome_distance(minpercentage,maxpercentage, filename_pattern):
    """
    Collect normalized distance between two replisomes,
    pooled over all trajectories.

    Returns:
        list: [distance]
    """
    print(f"--- Collecting distance data for {minpercentage} to {maxpercentage}% ---")

    frame = minpercentage / 100.0 * 2000
    minframe = int(frame)
    frame = maxpercentage / 100.0 * 2000
    maxframe = int(frame )

    numbers_to_skip = {}

    distances = []
    iters,tms=np.genfromtxt(f'{filename_pattern}/seginfo.txt',unpack=True)
    for i in iters:
        i=int(i)
        print(i)
        if i in numbers_to_skip:
            continue

        try:
            input_file = gsd.hoomd.open(f'{filename_pattern}/run_{i}_20k_3.50_fullringrepsoft.gsd', mode='r')
        except FileNotFoundError:
            print(f"Warning: File for run {i} not found. Skipping.")
            continue

        frame_counter = minframe
        j=minframe
        for frame in input_file[minframe:maxframe]:
            length = calculate_instantaneous_length(frame_counter, 28.0, 2000)
            typeid_indices = np.where(frame.particles.typeid == 4)[0]

            repindex1 = int(j/4)   
            repindex2 = 999- int(j/4)
            typeid_indices=[repindex1,repindex2]
            # Expect exactly two replisomes
            if len(typeid_indices) != 2:
                frame_counter += 1
                continue

            z_norm = []
            for pid in typeid_indices:
                z = normalize_coordinate_in_moving_system(
                    frame.particles.position[pid][2], length, 28.0
                )
                z_norm.append(z)

            # Distance between the two replisomes
            if len(z_norm)!=2:
                print(len(z_norm))
            dist = abs(z_norm[0] - z_norm[1])

            # Optional minimum-image distance
            # dist = min(dist, 1 - dist)

            # distances.append(dist*0.075)
            distances.append(dist)
            frame_counter += 1
            j=j+1
    print(f"Finished collecting distance data for {minpercentage} to {maxpercentage}%. "
          f"Found {len(distances)} data points.")
    return distances


# --- Main Plotting Script ---

if __name__ == "__main__":

    # percentages_to_plot = [[5,10],[10,20],[20,30],[70,80],[80,90]]
    percentages_to_plot = [[5,25],[70,90]]
    markers=['s','o']
    col=['red','blue']
    # filename_pattern = (
    #     '1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru1'
    # )
    filename_pattern = (
        '1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz'
    )
    plt.figure(figsize=(14,12))
    for p,m,c in zip(percentages_to_plot,markers,col):

        distances = collect_replisome_distance(p[0],p[1], filename_pattern)
        
        mean_dist = np.mean(distances)
        print(f"Mean replisome–replisome distance at {p}% = {mean_dist:.4f}")
        
        
        # plt.hist(
        #     distances,
        #     bins=20,
        #     # range=(0, 1),
        #     density=True,
        #     histtype='step',
        #     alpha=1,
        #     linewidth=3,
        #     label=f'{p[0]}% - {p[1]}% : ⟨d⟩ = {mean_dist:.3f}'
        # )

        prob,bins=np.histogram(
            distances,
            bins=15,
            range=(-0.1,1.05),
            density=True,
        )
        print(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        # plt.figure(figsize=(10, 8))
        plt.plot(bin_centers,prob,label=f'{p[0]}% - {p[1]}% : ⟨d⟩ = {mean_dist:.2f}',marker=m,color=c,markersize=12,mfc='none')
        plt.xlim(-0.1, 1.1)
        # plt.xlabel(r"d$\:\:(\mu m)$", fontsize=28 , fontweight='bold')
        plt.xlabel("d (a)", fontsize=36)
        plt.ylabel("Probability Density", fontsize=36)
        legend1 = plt.legend(loc='upper left', fontsize=14)
        plt.legend(fontsize=32)
        plt.tick_params(axis='x', pad=10)
        plt.tick_params(axis='y', pad=10)
        # plt.minorticks_on() 
        plt.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelsize=35,
            width=3,
            length=12
            )
        # plt.tick_params(axis='x', labelsize=24)
        # plt.tick_params(axis='y', labelsize=24)
        # for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        #     label.set_fontweight('bold')
        # plt.add_artist(legend1)  # keep both legends
        # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.title(f"Replisome–Replisome Distance Distribution", fontsize=28, fontweight='bold')
    # plt.savefig(f'{filename_pattern}/replisomedist_100ParA_cellsized+120_model2.pdf',dpi=300)
    plt.savefig(f'{filename_pattern}/replisomedist_100ParA_cellsized+120_model1.pdf',dpi=300)
    plt.show()




    