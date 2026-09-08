#__________________________________calculating segregation time using center of mass___________________________________________________

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
import os
A=5
addmon=20
file=open(f'1000monruns/70ParA/A5soft_20k_8.50_fullring_1000_70_28_4_10krate_assym_popz_extru/segregationtime1.txt',mode='w')
normseg=0
avgsegtime=0
repcom=True
for iter in range(30,60):
    os.makedirs(f'1000monruns/70ParA/A5soft_20k_8.50_fullring_1000_70_28_4_10krate_assym_popz_extru/graph', exist_ok=True)
    input=gsd.hoomd.open(f'1000monruns/70ParA/A5soft_20k_8.50_fullring_1000_70_28_4_10krate_assym_popz_extru/run_{iter}_20k_3.50_fullringrepsoft.gsd',mode='r')
    framenum = []
    comframe=[]
    com1=[]
    com2=[]
    framteiter=0
    isseg=False
    reptime=1
    repcom=True
    for frame in input:
        R=0
        N = frame.particles.N
        position = frame.particles.position
        R=N-1001
        CM1=np.zeros(3)
        CM2=np.zeros(3)
        count=0
        if(R>0):
            for k in range(3,R,2):
                l=(k-1)/2
                m=1000-(k-1)/2
                CM1=CM1+position[0]+position[int(l)]+position[int(m)]
                CM2=CM2+position[1000]+position[int(1000+2*l-1)]+position[int(1000+2*l)]
                count=count+3
            

            if(R==1000):
                CM1=CM1+position[500]
                CM2=CM2+position[1999]
                count=count+1
            # comframe.append(i)    
            CM1=CM1/count
            CM2=CM2/count
        
        if(R==1000 and repcom==True):
            # print(R)
            reptime=framteiter
            repcom=False

        com1.append(CM1[2])
        com2.append(CM2[2])
        segdist=np.linalg.norm(CM2[2]-CM1[2])
        halfcylinder=(28+28/1000*R)/2
        if(segdist>halfcylinder and isseg==False ):
            segtime=framteiter
            isseg=True

        framenum.append(framteiter)
        framteiter+=1
    if(isseg==True):
        file.write(f'{segtime}\n')
        print(f'segtime:{segtime}\n')
        avgsegtime=avgsegtime+segtime
        normseg=normseg+1
    else:
        file.write('none \n')
        print(f'segtime:not segregated\n')
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(framenum)/reptime,com1,label='COM1')
    plt.plot(np.array(framenum)/reptime,com2,label='COM2')
    plt.legend()
    plt.title(f'Adding new monomers after {addmon}K')
    plt.xlabel('Frames')
    plt.ylabel('Z position')
    # plt.show()
    plt.savefig(f'1000monruns/70ParA/A5soft_20k_8.50_fullring_1000_70_28_4_10krate_assym_popz_extru/graph/run_{iter}_{addmon}k_3.50_fullcom.png',dpi=300)
    plt.close()
avgsegtime=avgsegtime/normseg/reptime
file.write(f'Average segregation time = {avgsegtime} \n replication time in the form of number of frames={reptime}')
print(f'average segragation time = {avgsegtime}')
file.close()