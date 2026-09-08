import gsd.hoomd
import numpy as np 
import matplotlib.pyplot as plt
import os

directory='1000monruns/100ParA/expo/A5soft_20k_8.50_fullring_1000_100_28_4_10krate_assym_popz_extru1'
parA=100
os.makedirs(f'{directory}/locitraj',exist_ok=True)

L=28
initparticle=1000
freq=100
locipos=[0,-31,39,-88,86,-121,134,-178,54]
# locipos=[86]
# locipos=[1]
ntime = int(2400/freq)
maxiter=120

iters,tt=np.genfromtxt(f'{directory}/seginfo.txt',unpack=True)
# print(iters)
# iters=int(iters)
maxiter=len(iters[:])
lociindex=np.zeros(len(locipos))
for i,lo in enumerate(locipos):
    if lo>=0:
        lociindex[i]=int(lo*500/180)
    else:
        lociindex[i]=1000+int(lo*500/180)

# store all iterations
all_loci1=np.zeros((maxiter,len(locipos),ntime))
all_loci2=np.zeros((maxiter,len(locipos),ntime))
count=0


for iter in iters[:]:
# for iter in range(maxiter,240):
    iter=int(iter)
    # print(f'\r{iter}',end='')

    print(f'\r{iter}')
    
    try:
        file=gsd.hoomd.open(f'{directory}/run_{iter}_20k_3.50_fullringrepsoft.gsd',mode='r')
    except FileNotFoundError:
        print(f"Warning: File for run {i} not found. Skipping.")
        continue
    i=0
    k=0
    timefile1=np.zeros(ntime)
    loci1=np.zeros((len(locipos),ntime))
    loci2=np.zeros((len(locipos),ntime))

    for frames in file:

        if (k%freq==0):

            pos=frames.particles.position
            time=int(int(k/4)-1)

            lenth=L
            if time>=0 and time<=(initparticle/2):
                kk = (np.log(2*L) - np.log(L)) / (initparticle/2)
                lenth=L*np.exp(kk*time)

            if time>initparticle/2:
                lenth=2*L

            timefile1[i]=k*50/100/1000
            R=frames.particles.N
            d1=0
            d2=0
            d3=0
            d4=0
            for j in range(len(locipos)):

                if locipos[j]>=0:

                    if R<=1000+2*lociindex[j]+2:
                        d1=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d2=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d3=1-d1
                        d4=1-d2
                        if min(d1,d2,d3,d4)==d1 or min(d1,d2,d3,d4)==d2:
                            loci1[j,i]=min(d1,d2)
                            loci2[j,i]=max(d1,d2)
                        else:
                            loci1[j,i]=min(d3,d4)
                            loci2[j,i]=max(d3,d4)
                        # loci1[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        # loci2[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth

                    if R>1000+2*lociindex[j]+2:
                        d1=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d2=np.abs(pos[1000+2*int(lociindex[j])][2]-28)/lenth
                        d3=1-d1
                        d4=1-d2
                        if min(d1,d2,d3,d4)==d1 or min(d1,d2,d3,d4)==d2:
                            loci1[j,i]=min(d1,d2)
                            loci2[j,i]=max(d1,d2)
                        else:
                            loci1[j,i]=min(d3,d4)
                            loci2[j,i]=max(d3,d4)
                        # loci1[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        # loci2[j,i]=np.abs(pos[1000+2*int(lociindex[j])][2]-28)/lenth
                    


                else:

                    neglociindex=1000-lociindex[j]

                    if R<=1000+2*abs(neglociindex)+2:
                        d1=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d2=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d3=1-d1
                        d4=1-d2
                        if min(d1,d2,d3,d4)==d1 or min(d1,d2,d3,d4)==d2:
                            loci1[j,i]=min(d1,d2)
                            loci2[j,i]=max(d1,d2)
                        else:
                            loci1[j,i]=min(d3,d4)
                            loci2[j,i]=max(d3,d4)
                        # loci1[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        # loci2[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth

                    if R>1000+2*abs(neglociindex)+2:
                        d1=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        d2=np.abs(pos[int(1000+2*neglociindex+1)][2]-28)/lenth
                        d3=1-d1
                        d4=1-d2
                        if min(d1,d2,d3,d4)==d1 or min(d1,d2,d3,d4)==d2:
                            loci1[j,i]=min(d1,d2)
                            loci2[j,i]=max(d1,d2)
                        else:
                            loci1[j,i]=min(d3,d4)
                            loci2[j,i]=max(d3,d4)

                        # loci1[j,i]=np.abs(pos[int(lociindex[j])][2]-28)/lenth
                        # loci2[j,i]=np.abs(pos[int(1000+2*neglociindex+1)][2]-28)/lenth
                
            i=i+1
            k=k+1
        else:
            k=k+1

    all_loci1[count] = loci1
    all_loci2[count] = loci2
    count=count+1


# mean and std
mean_loci1 = np.mean(all_loci1,axis=0)
std_loci1  = np.std(all_loci1,axis=0)

mean_loci2 = np.mean(all_loci2,axis=0)
std_loci2  = np.std(all_loci2,axis=0)


for j in range(len(locipos)):

    fig, ax = plt.subplots(figsize=(12,10))

    ax.set_ylim(0,1)

    m1 = mean_loci1[j].copy()
    s1 = std_loci1[j].copy()

    m2 = mean_loci2[j].copy()
    s2 = std_loci2[j].copy()

    # flip if first value > 0.5
    # if m1[0] > 0.5:
    #     m1 = 1 - m1

    # if m2[0] > 0.5:
    #     m2 = 1 - m2


    # for k in range(len(m1)):

    #     if m1[k]>0.5:
    #         if m2[k]>0.5:
    #             print('testtest!!!!!!')
    #             l1= 1 - m1[k]
    #             l2= 1 - m2[k]


    #             m1[k] = l2
    #             m2[k] = l1


    #     if m1[k]<m2[k]:
    #         if (1-m2[k])<m1[k]:
    #             mflag=m1[k]
    #             m1[k]=m2[k]
    #             m2[k]=mflag
    #             sflag=s1[k]
    #             s1[k]=s2[k]
    #             s2[k]=sflag
    #     else:
    #         if (1-m1[k])<m2[k]:
    #             mflag=m1[k]
    #             m1[k]=m2[k]
    #             m2[k]=mflag
    #             sflag=s1[k]
    #             s1[k]=s2[k]
    #             s2[k]=sflag   

    ax.plot(timefile1, m1, label='copy 1',marker='s',linestyle='dashed')
    ax.fill_between(timefile1,
                    m1 - s1/2,
                    m1 + s1/2,
                    alpha=0.3)

    ax.plot(timefile1, m2, label='copy 2',marker='o',linestyle='dashed')
    ax.fill_between(timefile1,
                    m2 - s2/2,
                    m2 + s2/2,
                    alpha=0.3)

    ax.set_xlabel(r'$\tau\ (\tau_{\mathrm{replication}})$',
                  fontsize=50, fontweight='bold')
    ax.set_ylabel(r'$Z / L_{\mathrm{cyl}}$',
                  fontsize=50, fontweight='bold')
    # ax.legend()
    # ax.grid(True)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # ax.set_title(f"Mean loci position of {locipos[j]}$^\\circ$",
    #              fontweight='bold', fontsize=34)

    # ---------- inset circular chromosome ----------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    upper=[39,54,86,-31,-88]
    if locipos[j] in upper:
        axins = inset_axes(ax, width="40%", height="40%", loc='upper left')
    else:
        axins = inset_axes(ax, width="40%", height="40%", loc='lower left')
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=3)
    axins.add_patch(circle)

    # 0 degree at top
    axins.plot([0, 0], [0, 1], '--', linewidth=1)
    axins.text(0, 1.4, '0°', fontsize=28, ha='center', va='center')

    theta_deg = locipos[j]

    # 0 at top, clockwise positive
    theta = np.deg2rad(90 - theta_deg)

    x = np.cos(theta)
    y = np.sin(theta)

    axins.plot([0, x], [0, y], linewidth=2)
    axins.scatter(x, y, s=25)
    axins.text(x*1.6, y*1.6, f'{theta_deg}°',
               fontsize=28, ha='center', va='center')

    axins.set_xlim(-2, 2)
    axins.set_ylim(-2,2)
    axins.set_aspect('equal')
    axins.axis('off')
    # -----------------------------------------------

    plt.tight_layout()
    plot_data = {
        'time': timefile1,
        'mean_loci1': m1,
        'std_loci1': s1,
        'mean_loci2': m2,
        'std_loci2': s2,
        'loci_position': locipos[j]
    }

    np.save(
        f'{directory}/locitraj/100_loci_{locipos[j]}_sim_asexp.npy',
        plot_data,
        allow_pickle=True
    )
    # plt.savefig(
    #     f'{directory}/locitraj/100_loci_{locipos[j]}_simdata.pdf',
    #     dpi=300
    # )
    # plt.savefig(
    #     f'{directory}/locitraj/100_loci_{locipos[j]}_simdata.eps',
    #     dpi=300
    # )

    plt.close()
    # plt.show()