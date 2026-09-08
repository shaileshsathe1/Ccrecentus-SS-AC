import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def calculate_hic_map_mpi(trajectory_file, distance_cutoff, num_particles):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Use trajectory passed in
    traj = trajectory_file
    startingframe=2100
    endingframe=2399
    # Determine frame slicing
    total_frames = len(traj[startingframe:endingframe])
    frames_per_rank = total_frames // size
    extra_frames = total_frames % size

    if rank < extra_frames:
        start_frame = rank * (frames_per_rank + 1)
        end_frame = start_frame + frames_per_rank + 1
    else:
        start_frame = rank * frames_per_rank + extra_frames
        end_frame = start_frame + frames_per_rank

    # Initialize local Hi-C matrix
    local_hic_map = np.zeros((num_particles, num_particles), dtype=int)

    midpoint = 1000  # For 8000, this is 4000

    # Process frames assigned to this rank
    for frame_idx in range(start_frame + startingframe, end_frame + startingframe):
        print(f"Rank {rank} processing frame {frame_idx}")
        frame = traj[frame_idx]
        positions = frame.particles.position

        for i in range(num_particles - 1):
            for j in range(i + 1, num_particles):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < distance_cutoff:
                    if i >= midpoint and j < midpoint:
                        k = i - midpoint
                        if k % 2 == 0:
                            i1 = num_particles - k // 2 - 1
                        else:
                            i1 = midpoint + k // 2
                        local_hic_map[i1, j] += 1
                        local_hic_map[j, i1] += 1
                    elif i < midpoint and j >= midpoint:
                        k = j - midpoint
                        if k % 2 == 0:
                            j1 = num_particles - k // 2 - 1
                        else:
                            j1 = midpoint + k // 2
                        local_hic_map[i, j1] += 1
                        local_hic_map[j1, i] += 1
                    elif i >= midpoint and j >= midpoint:
                        ki = i - midpoint
                        if ki % 2 == 0:
                            i1 = num_particles - ki // 2 - 1
                        else:
                            i1 = midpoint + ki // 2
                        kj = j - midpoint
                        if kj % 2 == 0:
                            j1 = num_particles - kj // 2 - 1
                        else:
                            j1 = midpoint + kj // 2
                        local_hic_map[i1, j1] += 1
                        local_hic_map[j1, i1] += 1
                    else:
                        local_hic_map[i, j] += 1
                        local_hic_map[j, i] += 1

    # Gather results to rank 0
    global_hic_map = np.zeros((num_particles, num_particles), dtype=int)
    comm.Reduce(local_hic_map, global_hic_map, op=MPI.SUM, root=0)

    # Plot on root process
    # if rank == 0:
    #     plt.imshow(global_hic_map, cmap='rainbow', interpolation='nearest')
    #     plt.colorbar()
    #     plt.title('Hi-C Contact Map')
    #     plt.xlabel('Particle Index')
    #     plt.ylabel('Particle Index')
    #     plt.show()

        # Plot on root process
    if rank == 0:
        # Add a small epsilon instead of +1
        eps = 1e-6

        log_hic_map = np.log10(global_hic_map + eps)
       
        plt.figure(figsize=(20,20))    
        # plt.imshow(log_hic_map, cmap='Blues', interpolation='nearest',origin='lower')
        plt.imshow(global_hic_map, cmap='Blues', interpolation='nearest',origin='lower')
        cbar = plt.colorbar()
        cbar.set_label('log10(Contact Frequency + ε)')
        cbar.set_label('Contact Frequency')
        cbar.mappable.set_clim(vmin=0, vmax=0.2)
        # plt.title('Hi-C Contact Map (log scale)', fontsize=24)
        plt.title('Hi-C Contact Map', fontsize=24)
        plt.xlabel('Particle Index', fontsize=20)
        plt.ylabel('Particle Index', fontsize=20)
        plt.savefig('1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru100_HiC_Nonlog1.pdf',dpi=500)
        plt.show()


# ===== Run the code below with mpirun/mpiexec =====

trajfile = gsd.hoomd.open('1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru1/run_101_20k_3.50_fullringrepsoft.gsd', mode='r')
distance_cutoff =3
num_particles = 2000

calculate_hic_map_mpi(trajfile, distance_cutoff, num_particles)
trajfile.close()
