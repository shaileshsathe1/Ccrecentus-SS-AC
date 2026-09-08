import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

L = 28
initparticle = 1000
maxiter =229
startiter =228
nframes =2400


traj_paths = [
    # "1000monruns/130ParA/expo/A5soft_20k_8.50_fullring_1000_130_28_4_10krate_assym_popz_ParAchange",
    # "1000monruns/100ParA/expo/entropycheck/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_selfexclusion_extru",
    # "1000monruns/160ParA/A5soft_20k_8.50_fullring_1000_160_28_4_10krate_assym_popz_extru",
    # "1000monruns/160ParA/expo/A5soft_20k_8.50_fullring_1000_160_28_4_10krate_assym_popz",
    # "1000monruns/70ParA/expo/A5soft_20k_8.50_fullring_1000_70_28_4_10krate_assym_popz_extru",
    # '1000monruns/100ParA/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru'
    # '1000monruns/100ParA/Uniform/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru2',
    '1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru1',
    # '1000monruns/160ParA/expo/A5soft_20k_8.50_fullring_1000_160_28_4_10krate_assym_popz_extru2'
    # '1000monruns/Ringreplication/A5soft_20k_8.50_fullring_1000_70_28_4_assym_popz_extru'
    # '1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz'
]

case=0
for path in traj_paths:
    os.makedirs(f'{path}/traj',exist_ok=True)
    # infofile=open(f'{path}/seginfo.txt','w')
    avgdist = np.zeros(nframes)
    timefile = np.zeros(nframes)
    avgpasstime=0
    count=0
    count1=0
    for iter in range(startiter, maxiter):
        ori1=np.zeros(nframes)
        ori2=np.zeros(nframes)
        oridist = []
        filepath = Path(path) / f"run_{iter}_20k_3.50_fullringrepsoft.gsd"

        if filepath.exists():
            count1=count1+1
            data = gsd.hoomd.open(f'{path}/run_{iter}_20k_3.50_fullringrepsoft.gsd', mode='r')
            if len(data)==nframes:
                checkflag=True
                i = 0
                for frame in data[:nframes]:

                    if iter == startiter:
                        timefile[i] = i * 50 / 100 / 1000

                    R = frame.particles.N
                    time = int(int(i/4) - 1)
                    length = L
                    pospopz=frame.particles.position[(R-1)][2]

                    
                    if 0 <= time <= (initparticle/2):
                        k = (np.log(2*L) - np.log(L)) / (initparticle/2)
                        length = L * np.exp(k*time)
                    elif time > initparticle/2:
                        length = 2*L

                    if R > 1001:
                        dist = (frame.particles.position[0][2] -
                                frame.particles.position[1000][2]) / length
                        posori=frame.particles.position[1000][2]
                        ori1[i]=(28-frame.particles.position[0][2]) / length
                        ori2[i]=(28-frame.particles.position[1000][2]) / length
                    else:
                        dist = 0
                        ori1[i]=frame.particles.position[0][2]
                        ori2[i]=frame.particles.position[0][2]

                    oridist.append(abs(dist))
                    
                    checkdist=abs(pospopz-posori)
                    if checkflag==True and checkdist<1.0:
                        passsagetime= timefile[i]
                        checkflag=False
                        avgpasstime+=passsagetime
                        count=count+1
                    i += 1
                
                
                avgdist += np.array(oridist)
            
                
                if checkflag==False:
                    print( f'passage time for {iter} = {np.mean(passsagetime)}')
                    lab=f'Passage time ={passsagetime}'
                    # infofile.write(f'{iter} {passsagetime} \n')
                else:
                    # print(f'no segregation')
                    lab=f'Passage time =none'
                case +=1 
                # plt.figure(figsize=(12,10))
                # plt.plot(timefile, oridist, markersize=4,
                #     markerfacecolor='none', label=lab)
                # plt.legend(fontsize=22)
                # plt.title("Distance between Two Oris", fontweight='bold', fontsize=28)
                # plt.xlabel(r'$\tau\ (\tau_{\mathrm{replication}})$', fontsize=26, fontweight='bold')
                # plt.ylabel(r'$\Delta Z / L_{\mathrm{cyl}}$', fontsize=26, fontweight='bold')
                # plt.tick_params(axis='both', labelsize=18)

                # for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                #     label.set_fontweight('bold')

                # plt.grid(True)
                # plt.tight_layout()
                # plt.savefig(f'{path}/traj/run_{iter}_oritraj.png',dpi=300)
                # plt.close()

                plt.figure(figsize=(12,10))
                plt.plot(timefile, ori1, markersize=4,
                    markerfacecolor='none',label='Ori-1',color='red')
                plt.plot(timefile, ori2, markersize=4,
                    markerfacecolor='none',label='Ori-2',color='blue')
                plt.legend(fontsize=24)
                # plt.title("Ori Trajectory", fontweight='bold', fontsize=30)
                plt.xlabel(r'$\tau\ (\tau_{\mathrm{rep}})$', fontsize=30, fontweight='bold')
                # plt.ylabel(r'$z_{\textit{oriC}} / L_{\mathrm{cyl}}$', fontsize=28,fontweight='bold')
                plt.ylabel(r'$ z_{{oriC}} \: \:/ \: \:L_{\mathrm{cyl}}$', fontsize=30, fontweight='bold')
                plt.tick_params(axis='both', labelsize=24)

                for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                    label.set_fontweight('bold')

                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{path}/traj/run_{iter}_oritraj1.pdf',dpi=300)
                plt.close()
            else:
                print(f"File missing for iteration {iter}. Skipping.")
                continue
        else:
            print(f"File missing for iteration {iter}. Skipping.")
            continue
    avgdist /= (count1)
    if count!=0:
        avgpasstime=avgpasstime/count
        print(f'average Passage time={avgpasstime}')
    # infofile.close()
    # datafile=np.save(f'{path}/traj/240avgori.npy',(timefile,avgdist))
    plt.figure(figsize=(12,10))
    plt.plot(timefile, avgdist, markersize=4,
            markerfacecolor='none', label=f'Passage time ={avgpasstime}')
    plt.legend(fontsize=22)
    plt.title("Distance between Two Oris", fontweight='bold', fontsize=28)
    plt.xlabel(r'$\tau\ (\tau_{\mathrm{replication}})$', fontsize=32, fontweight='bold')
    plt.ylabel(r'$\Delta Z / L_{\mathrm{cyl}}$', fontsize=28, fontweight='bold')
    plt.tick_params(axis='both', labelsize=18)

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path}/traj/oritraj.png',dpi=300)
    plt.close()