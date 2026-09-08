import hoomd
import gsd.hoomd
import os
import matplotlib.pyplot as plt
import hoomd.snapshot
import hoomd.write.gsd
import numpy as np
import time

start_time = time.time()


#----------------------------------------------PARAMETERS-------------------------------------------------------------------------------------

initparticles=1000
epsilon = 1
KT = 1.0
L=28
A=5
ParAnum=160
addmon=2e4
equiltime=2*addmon
ParArate=int(addmon/2)
directory=f'1000monruns/{ParAnum}ParA/expo/A{A}soft_{int(addmon/1000)}k_8.50_fullring_1000_{ParAnum}_28_4_{int(ParArate/1000)}krate_assym_popz_extru1'
# directory1=f'1000monruns/100ParA/expo/A{A}soft_{int(addmon/1000)}k_8.50_fullring_1000_100_28_4_{int(ParArate/1000)}krate_assym_popz_extru1/initconfig'
os.makedirs(directory,exist_ok=True)
# os.makedirs(directory1,exist_ok=True)

def writeinitconf():

    file=open('1000monruns/{ParAnum}ParA/expo/initconf.txt',mode='w')
    file.write(f'A={A} \n addmon={addmon} \n ParA rate={ParArate} L={L} \n R={4} \n initparticles={initparticles} ')
    file.write(f'\n initfile={inputfile} \n Directory={directory}')
    file.close()

inputfile=f'initfiles/equiled_1000_28_4_crosslinked1.gsd'
writeinitconf()

# vv=15*7
# for iter in range(vv,vv+15):
for iter in range(240,241):


    #-----------------------------------------------INPUT AND OUTPUT FILES-------------------------------------------------



    


    output = f'run_{iter}_{int(addmon/1000)}k_3.50_fullringrepsoft.gsd'
    # innput=f'run_{iter}_{int(addmon/1000)}k_3.50_fullinitconf.gsd'
    output=os.path.join(directory,output)
    # innput=os.path.join(directory1,innput)
    try:
        if os.path.exists(output):
            os.remove(output)
        # if os.path.exists(innput):
        #     os.remove(innput)  
    except FileNotFoundError:
        pass

    # file=gsd.hoomd.open(output,mode='w')
    # file.close()
    inputfile=inputfile

    writeinitconf()

    #----------------------------------------------CUSTOM WRITER---------------------------------------------------------------------

    
        



    class CustomWriteFrames(hoomd.custom.Action):
        def __init__(self, filename):
            self.filename = filename
            self.gsd_file = gsd.hoomd.open(name=filename, mode='w')
            self.frames_written = 0

        def act(self, timestep):
            snapshot = self._state.get_snapshot()
            if snapshot.communicator.rank == 0:
                frame = gsd.hoomd.Frame()
                frame.configuration.step = timestep
                frame.configuration.box = snapshot.configuration.box
                frame.particles.N = snapshot.particles.N
                frame.particles.types = snapshot.particles.types
                frame.particles.typeid = snapshot.particles.typeid
                frame.particles.position = snapshot.particles.position
                frame.bonds.N = snapshot.bonds.N
                frame.bonds.types = snapshot.bonds.types
                frame.bonds.typeid = snapshot.bonds.typeid
                frame.bonds.group = snapshot.bonds.group
                frame.particles.velocity=snapshot.particles.velocity
                # Validate frame before writing
                assert frame.particles.position.shape[0] == frame.particles.N, \
                    f"Position count {frame.particles.position.shape[0]} does not match particle count {frame.particles.N}"
                assert frame.bonds.group.shape[0] == frame.bonds.N, \
                    f"Bond group count {frame.bonds.group.shape[0]} does not match bond count {frame.bonds.N}"

                # Write the frame to the GSD file
                self.gsd_file.append(frame)
                self.frames_written += 1
                print(f"\rFrame {self.frames_written} written at timestep {timestep}", end=' ')

        def close(self):

            self.gsd_file.close()

    #-----------------------------------SOFT TUNER---------------------------------------------------------------------------------------------


    class softtuner(hoomd.custom.Action):
        def __init__(self,pair_force,initA,finalA,rate):
            self.pair_force=pair_force
            self.k=initA
            self.finalK=finalA
            self.rate=rate
        def act(self, timestep):
            if self.k < self.finalK:
                self.k += self.rate
                self.pair_force.params[('replisome', 'A')] = dict(A=self.k, rho=0.6 , C=0)
                self.pair_force.params[('replisome', 'ParB')] = dict(A=self.k, rho=0.6, C=0)
                self.pair_force.params[('replisome', 'ParA')] = dict(A=self.k, rho=0.6, C=0)
                self.pair_force.params[('replisome', 'replisome')] = dict(A=self.k, rho=0.6, C=0)
                self.pair_force.params[('replisome', 'D2')] = dict(A=self.k, rho=0.6, C=0)



    #---------------------------------------BOND TUNER-------------------------------------------------------------------------------------------------


    class BondTuner(hoomd.custom.Action):
        def __init__(self,bond_force,initK,finalK,rate):
            self.bond_force=bond_force
            self.k=initK
            self.finalK=finalK
            self.rate=rate
        def act(self, timestep):
            if self.k < self.finalK:
                self.k += self.rate
                self.bond_force.params['teatheredbond'] = dict(k=self.k, r0=1)

                # print(f"\rupdated bond k to {self.k}",end='')


    
    #--------------------------------------------CHANGE OF VARIABLES----------------------------------------------------------------------------------


    cartesian_to_cylindrical = lambda x, y, z: [np.sqrt(x**2 + y**2),np.arctan2(y, x), z]
    cylindrical_to_cartesian = lambda r, theta, z: [r *np.cos(theta), r * np.sin(theta), z]



#---------------------------------------REPLICATION------------------------------------------------------------------------------------------------



    class Replication(hoomd.custom.Action):
        def __init__(self,customsoft,custombond,sim_object,state,walls_potential,addingrate,extrusion_rate,initparticle,equiltime,L):
            self.state=state
            self.soft=customsoft
            self.bond=custombond
            self.sim=sim_object
            self.wall=walls_potential
            self.initparticle=initparticle
            self.isnap=self.state.get_snapshot()
            self.addmon=addingrate
            self.extru_rate=extrusion_rate
            self.i=0
            self.equiltime=equiltime
            self.box=self.isnap.configuration.box
            self.L=L
            self.dl=0
            self.crosslinks=[]

        def ival(self,timestep):
            forkpos=(timestep-self.equiltime)/self.addmon
            return forkpos


        def wall_shift(self,timestep):
            self.i=int(self.ival(timestep)-1)
            lenth=self.L
            # if self.sim.device.communicator.rank==0:
            #     print('initL=', self.L)

            #linear growth

            # if self.i==0:
            #     self.dl=17.5/500
            # elif self.i==250:
            #     self.dl=17.5/500
            # elif self.i>0 and self.i<250:
            #     self.dl=35/500
            # else:
            #     self.dl=0
            # self.L=self.L+self.dl

            #for exponential growth
            if self.i>=0 and self.i<=(self.initparticle/2):
                k = (np.log(2*self.L) - np.log(self.L)) / (self.initparticle/2)
                lenth=self.L*np.exp(k*self.i)
            if self.i>self.initparticle/2:
                lenth=2*self.L



            top=hoomd.wall.Plane(origin=(0,0,self.L),normal=(0,0,-1))
            bottom=hoomd.wall.Plane(origin=(0,0,self.L-lenth),normal=(0,0,1))

            
            return top,bottom
                
        def updateposition(self,timestep):
            self.isnap=self.state.get_snapshot()
            positions=[]
            ptypeid=[]
            vel=[]
            ptypes=[]
            Nlen=0
            if self.isnap.communicator.rank==0:
                self.i=int(self.ival(timestep)-1)
                positions=self.isnap.particles.position
                ptypeid = self.isnap.particles.typeid
                vel=self.isnap.particles.velocity
                ptypes=self.isnap.particles.types

                if self.sim.device.communicator.rank==0:
                    print(f"\rrreplication stage={int(self.i/(self.initparticle/2)*100)}%  i={self.i}",end=' ')#
                    
                
                if self.i<=(self.initparticle/2) and self.i>=0:
                    positions=self.isnap.particles.position[:-2]
                    ptypeid = self.isnap.particles.typeid[:-2]
                    vel=self.isnap.particles.velocity[:-2]
                    ptypes=self.isnap.particles.types
                    softtune=softtuner(self.sim.operations.integrator.forces[3],0,A,0.5)
                    self.sim.operations.updaters.remove(self.soft)
                    self.soft=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(50),action=softtune)
                    self.sim.operations.updaters.append(self.soft)

                    if self.i <=2:
                        ptypeid[self.initparticle:]=1
                        pos1=positions[self.i]
                        pos2=positions[self.initparticle - (self.i+1)]
                        pos1=cartesian_to_cylindrical(pos1[0],pos1[1],pos1[2])
                        pos2=cartesian_to_cylindrical(pos2[0],pos2[1],pos2[2])
                        if (pos1[0]>1):
                            pos1[0]=np.absolute(pos1[0]-1)
                        else:
                            pos1[0]=0
                        if (pos2[0]>1):
                            pos2[0]=np.absolute(pos2[0]-1)
                        else:
                            pos2[0]=0
                        pos1=cylindrical_to_cartesian(pos1[0],pos1[1],pos1[2])
                        pos2=cylindrical_to_cartesian(pos2[0],pos2[1],pos2[2])
                        new_particle_position=[ np.array(pos1), np.array(pos2)]
                        positions = np.vstack((positions, new_particle_position))
                        ptypeid = np.append(ptypeid, [4, 4])
                        newvel=[[0,0,0],[0,0,0]]
                        vel=np.append(vel,newvel)
                        
                    
                    elif self.i >2 and self.i <(self.initparticle/2) :
                        changetype=np.where(ptypeid==4)[0]
                        for ids in changetype:
                            ptypeid[ids]=5
                        pos1=positions[self.i]
                        pos2=positions[self.initparticle - (self.i+1)]
                        pos1=cartesian_to_cylindrical(pos1[0],pos1[1],pos1[2])
                        pos2=cartesian_to_cylindrical(pos2[0],pos2[1],pos2[2])
                        if (pos1[0]>1):
                            pos1[0]=np.absolute(pos1[0]-1)
                        else:
                            pos1[0]=0
                        if (pos2[0]>1):
                            pos2[0]=np.absolute(pos2[0]-1)
                        else:
                            pos2[0]=0
                        pos1=cylindrical_to_cartesian(pos1[0],pos1[1],pos1[2])
                        pos2=cylindrical_to_cartesian(pos2[0],pos2[1],pos2[2])
                        new_particle_position=[ np.array(pos1), np.array(pos2)]
                        positions = np.vstack((positions, new_particle_position))
                        ptypeid = np.append(ptypeid, [4, 4])
                        newvel=[[0,0,0],[0,0,0]]
                        vel=np.append(vel,newvel)
                        
                    elif self.i== (self.initparticle/2):
                        changetype=np.where(ptypeid==4)[0]
                        for ids in changetype:
                            ptypeid[ids]=5
                        # print(zpos)
                        

                    

                    Nlen=len(positions)

                    k = (np.log(2*self.L) - np.log(self.L)) / (self.initparticle/2)
                    zpos=self.L*np.exp(k*self.i)
                    newpos=29-zpos
                    positions=np.vstack((positions,[[0,0,27],[0,0,newpos]]))

                    ptypeid = np.append(ptypeid, [2,6])
                    newvel=[[0,0,0],[0,0,0]]
                    vel=np.append(vel,newvel)

                    if self.sim.device.communicator.rank==0:
                        final_count = np.sum(ptypeid == 3)
                        print(f"\rParA count : {final_count}",end=' ')

            return positions,ptypeid,vel,ptypes,Nlen
        


        def updatebonds(self,timestep,Nlen):
            self.isnap=self.state.get_snapshot()
            self.i=int(self.ival(timestep)-1)
            bgroup=[]
            btypeid=[]
            btypes=[]
            if self.isnap.communicator.rank==0:
                bgroup=self.isnap.bonds.group
                btypeid=self.isnap.bonds.typeid
                btypes=self.isnap.bonds.types
                if self.i<(self.initparticle/2) and self.i>=0:
                    bondtune=BondTuner(self.sim.operations.integrator.forces[2],0,100,1)
                    self.sim.operations.updaters.remove(self.bond)
                    self.bond=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(100),action=bondtune)
                    self.sim.operations.updaters.append(self.bond)
                    bgroup=self.isnap.bonds.group[:-1]
                    btypeid=self.isnap.bonds.typeid[:-1]
                    btypes=self.isnap.bonds.types
                    new_bonds=[]
                    new_btypeid=[]
                    crossi=150

                    if self.i==0:
                        new_bonds = [
                            [self.initparticle,self.initparticle+1],
                            [1, self.initparticle],
                            [self.initparticle - 2, self.initparticle+1]
                        ]
                        new_btypeid = [0, 0, 0]
                        bgroup = np.vstack((bgroup, new_bonds))
                        btypeid = np.append(btypeid, new_btypeid)
                    elif self.i==int(self.initparticle/2-1):
                        bgroup = bgroup[:-2]
                        new_bonds=[
                            [self.initparticle+(2*self.i-2),self.initparticle+(2*self.i)],
                            [self.initparticle+(2*self.i-1),self.initparticle+(2*self.i+1)],
                            [self.initparticle+(2*self.i),self.initparticle+(2*self.i+1)],
                            [self.initparticle+(2*self.i+1),(self.i+1)]
                        ]
                        new_btypeid = [0,0]
                        bgroup = np.vstack((bgroup, new_bonds))
                        btypeid = np.append(btypeid, new_btypeid)
                    
                    else:
                        bgroup = bgroup[:-2]
                        new_bonds=[
                            [self.initparticle+(2*self.i-2),self.initparticle+(2*self.i)],
                            [self.initparticle+(2*self.i-1),self.initparticle+(2*self.i+1)],
                            [self.initparticle+2*self.i,(self.i+1)],
                            [self.initparticle+(2*self.i+1),self.initparticle-(self.i+2)]
                        ]
                        new_btypeid = [0, 0]
                        bgroup = np.vstack((bgroup, new_bonds))
                        btypeid = np.append(btypeid, new_btypeid)

                    
                    
                    bgroup=np.vstack((bgroup,[0,Nlen]))
                    btypeid=np.append(btypeid,[2])
                    

            return bgroup,btypeid,btypes


        


        def create_new_snap(self,timestep):
            positions,ptypeid,vel,ptypes,Nlen=self.updateposition(timestep=timestep)
            bgroup,btypeid,btypes=self.updatebonds(timestep,Nlen)
            i=self.ival(timestep)


            frame=gsd.hoomd.Frame()
            frame = gsd.hoomd.Frame()
            frame.configuration.box = self.box
            frame.particles.N = len(positions)
            frame.particles.types = ptypes
            frame.particles.typeid = ptypeid
            frame.particles.position = positions
            frame.bonds.N = len(bgroup)
            frame.bonds.types = btypes
            frame.bonds.typeid = btypeid
            frame.bonds.group = bgroup
            frame.particles.velocity=vel
            fsnapshot=hoomd.Snapshot.from_gsd_frame(frame,hoomd.communicator.Communicator())
            return fsnapshot
        

        
        

        def act(self,timestep):
            fsnap=self.create_new_snap(timestep)
            self.state.set_snapshot(fsnap)
            top,bottom=self.wall_shift(timestep)
            self.wall.walls[0] = top
            self.wall.walls[1] = bottom

#----------------------------------ParA redistribution---------------------------------------------------------------

    class ParAredist(hoomd.custom.Action):
        def __init__(self,sim_object,state,equiltime,neighboulist,addmon,L,N):
            self.sim=sim_object
            self.equiltime=equiltime
            self.state=state
            self.initsnap=self.state.get_snapshot()
            self.nlist=neighboulist
            self.addmon=addmon
            self.L=L
            self.N=N
            
        def ival(self,timestep):
            forkpos=(timestep-self.equiltime)/self.addmon
            return forkpos
        

        # def exp_dist(self,x,rho):
        #     norm=1/(rho*(1-np.exp(-11/rho)))
        #     func=norm*np.exp(-x/rho)
        #     return func
        
        def exp_dist(self,rho):

            decay_const=2.59
            max_particles=54

            parA=np.zeros(11)

            for num in range(11):
                parnum = int(np.exp(-num / decay_const) * max_particles)
                parA[num]=parnum/ParAnum
            return parA


        def wall_shift(self,timestep):
            self.i=int(self.ival(timestep)-1)
            lenth=self.L
            # if self.sim.device.communicator.rank==0:
            #     print('initL=', self.L)

            #linear growth

            # if self.i==0:
            #     self.dl=17.5/500
            # elif self.i==250:
            #     self.dl=17.5/500
            # elif self.i>0 and self.i<250:
            #     self.dl=35/500
            # else:
            #     self.dl=0
            # self.L=self.L+self.dl

            #for exponential growth
            if self.i>=0 and self.i<=int(self.N/2):
                k = (np.log(2*self.L) - np.log(self.L)) /(self.N/2)
                lenth=self.L*np.exp(k*self.i)
            if self.i>int(self.N/2):
                lenth=2*self.L
            return lenth

        def newstate(self,timestep):
            self.i=int(self.ival(timestep)-1)
            oldstate=self.state.get_snapshot()
            celllenth=self.wall_shift(timestep)
            leftedge=(self.L-celllenth)
            ptypeid=[]
            positions=[]
            ParB_ParA=[]
            daughters=[]
            parB_index=[]
            parA_index=[]
            lenghtindex=[[],[],[],[],[],[],[],[],[],[],[]]
            checkpopz=True
            decay_const=2.59
            max_particles=34
            sum=0

            # parA=np.zeros(11)

            # for num in range(11):
            #     parnum = int(np.exp(-num / decay_const) * max_particles)
            #     sum=sum+parnum
            #     parA[num]=parnum/100
            
            if oldstate.communicator.rank==0:

                ptypeid=oldstate.particles.typeid
                positions=oldstate.particles.position
                parB_index=np.where(oldstate.particles.typeid==1)[0]
                parA_index=np.where(oldstate.particles.typeid==3)[0]

                zpos=[]
                for n in parA_index:
                    zpos.append(positions[n][2])

                
                dl=((28-leftedge-2)/11)

                bins=[leftedge,leftedge+dl/2,leftedge+3*dl/2,leftedge+5*dl/2,leftedge+7*dl/2,leftedge+9*dl/2,
                      leftedge+11*dl/2,leftedge+13*dl/2,leftedge+15*dl/2,leftedge+17*dl/2,leftedge+19*dl/2,leftedge+21*dl/2]
                
                # print(bins)
                # bins=[0+1.25,2.5+1.25,5+1.25,7.5+1.25,10+1.25,12.5+1.25,15+1.25,17.5+1.25,20+1.25,22.5+1.25,25+1.25,27.5+1.25]
                prob=np.histogram(zpos,bins=bins)
                simprob=prob[0]/ParAnum
                # print(len(self.parA_index))
                # if (len((parA_index))!=64):
                #     print(len(parA_index))
                # print(len(self.parB_index))

                for parbidx in parB_index:
                    for paraidx in parA_index:
                        if paraidx not in ParB_ParA:
                            dist= np.linalg.norm((positions[parbidx]-positions[paraidx]))
                            if dist<2 :
                                ParB_ParA.append(paraidx)
                            # parA_index=np.delete(parA_index,np.where(parA_index==paraidx)[0])
                # daughter=np.where(ptypeid==5)[0]
                # self.normal_index=np.append(self.normal_index,daughter)
                
                # self.normal_index=np.where((ptypeid==0) | (ptypeid==5))[0] 
                daughters=np.where((ptypeid==0) | (ptypeid==5))[0]
                
                # print(self.normal_index)
                for i_dau in daughters:
                    pos=positions[i_dau][2]
                    index=int((pos-leftedge)//dl)
                    if index<=10:
                        lenghtindex[int(index)].append(i_dau)
                
                if np.linalg.norm((positions[-1]-positions[1000]))<1.0:

                    print('Popcheck hit!!!!!')
                    checkpopz=False
                else:
                    checkpopz=True
                # print(len(ParB_ParA))
                if checkpopz==True:
                    for PA in ParB_ParA:
                        if PA<self.N:
                            # print('check1')
                            ptypeid[PA]=0
                        else:
                            ptypeid[PA]=5
                        checknum=0
                        while True:
                            
                            # k=self.exp_dist(randint,1000)
                            pos=positions[PA][2]
                            indx=int((pos-leftedge)//dl)
                            # print(len(simprob),len(self.exp_dist(2.59)))
                            probdiff=self.exp_dist(2.59)-simprob

                            # print (f'{self.exp_dist(2.59)}\n,{simprob}\n,{probdiff}')
                            if(max(probdiff)==0):
                                scalingfactor=-1
                            else:
                                scalingfactor=1/max(probdiff)
                            
                            probdiff=probdiff*scalingfactor
                            randint=np.random.randint(0,11)
                            k=probdiff[randint]
                            randcheck=np.random.random()
                            if (k > randcheck and len(lenghtindex[randint])!=0):
                                idxflag=np.random.choice(lenghtindex[randint],size=1, replace=False)
                                lenghtindex[randint]=np.delete(lenghtindex[randint],np.where(lenghtindex[randint]==idxflag)[0])
                                ptypeid[idxflag]=3
                                # print('updated')
                                break
                            else:
                                print(f'\rstucked ', end=' ')
                                checknum=checknum+1
                                if checknum>1000:
                                    while True:
                                        randint=np.random.randint(0,11)
                                        if(len(lenghtindex[randint])!=0):
                                            idxflag=np.random.choice(lenghtindex[randint],size=1, replace=False)
                                            lenghtindex[randint]=np.delete(lenghtindex[randint],np.where(lenghtindex[randint]==idxflag)[0])
                                            ptypeid[idxflag]=3
                                            break
                                        else:
                                            continue
                                    break
                                continue
                else:
                    for PA in ParB_ParA:
                        if PA<self.N:
                            # print('check1')
                            ptypeid[PA]=0
                        else:
                            ptypeid[PA]=5
                        checknum=0
                        while True:
                            
                            # k=self.exp_dist(randint,1000)
                            pos=positions[PA][2]
                            indx=int((pos-leftedge)//dl)
                            # print(len(simprob),len(self.exp_dist(2.59)))
                            probdiff=self.exp_dist(2.59)-simprob

                            # print (f'{self.exp_dist(2.59)}\n,{simprob}\n,{probdiff}')
                            if(max(probdiff)==0):
                                scalingfactor=-1
                            else:
                                scalingfactor=1/max(probdiff)
                            
                            probdiff=probdiff*scalingfactor
                            randint=np.random.randint(0,11)
                            k=1
                            randcheck=np.random.random()
                            if (k > randcheck and len(lenghtindex[randint])!=0):
                                idxflag=np.random.choice(lenghtindex[randint],size=1, replace=False)
                                lenghtindex[randint]=np.delete(lenghtindex[randint],np.where(lenghtindex[randint]==idxflag)[0])
                                ptypeid[idxflag]=3
                                # print('updated')
                                break
                            else:
                                print(f'\rstucked ', end=' ')
                                checknum=checknum+1
                                if checknum>1000:
                                    while True:
                                        randint=np.random.randint(0,11)
                                        if(len(lenghtindex[randint])!=0):
                                            idxflag=np.random.choice(lenghtindex[randint],size=1, replace=False)
                                            lenghtindex[randint]=np.delete(lenghtindex[randint],np.where(lenghtindex[randint]==idxflag)[0])
                                            ptypeid[idxflag]=3
                                            break
                                        else:
                                            continue
                                    break
                                continue
                # success_count = 0
                # for _ in range(len(ParB_ParA)):
                #     for attempt in range(100):  # Try 100 times max
                #         randint = np.random.randint(0, 8)
                #         k = self.exp_dist(randint, 2.2)
                #         randcheck = np.random.random()

                #         if k > randcheck and len(lenghtindex[randint]) > 0:
                #             idxflag = np.random.choice(lenghtindex[randint])
                #             lenghtindex[randint] = np.delete(lenghtindex[randint], np.where(lenghtindex[randint] == idxflag)[0])
                #             ptypeid[idxflag] = 3
                #             success_count += 1
                #             break
                #     else:
                #         print("⚠️ Failed to find a replacement for one ParA.")

                # print(f"✅ Total ParA removed: {len(ParB_ParA)}, ParA added: {success_count}")

                
                # print(len(ParB_ParA))
                # for PA in ParB_ParA:
                #     if PA<500:
                #         # print('check1')
                #         ptypeid[PA]=0
                #     else:
                #         ptypeid[PA]=5
                # Count how many ParAs now exist
                final_count = np.sum(ptypeid == 3)
                # print(f"ParA count after update: {final_count}")

            return ptypeid
        
        def create_snap(self,timestep):
            ptypeid=self.newstate(timestep)
            oldstate=self.state.get_snapshot()
            frame = gsd.hoomd.Frame()
            if oldstate.communicator.rank==0:
                frame.configuration.box = oldstate.configuration.box
                frame.particles.N = oldstate.particles.N
                frame.particles.types = oldstate.particles.types
                frame.particles.typeid = ptypeid
                frame.particles.position = oldstate.particles.position
                frame.bonds.N = oldstate.bonds.N
                frame.bonds.types = oldstate.bonds.types
                frame.bonds.typeid = oldstate.bonds.typeid
                frame.bonds.group = oldstate.bonds.group
                frame.particles.velocity=oldstate.particles.velocity
            fsnapshot=hoomd.Snapshot.from_gsd_frame(frame,hoomd.communicator.Communicator())
            return fsnapshot
        
        def act(self, timestep):
            fsnap=self.create_snap(timestep)
            self.state.set_snapshot(fsnap)







#----------------------------------initstate modification -----------------------------------------------

    class initmod(hoomd.custom.Action):
        def __init__(self,state,equiltime,L,N,input):
            self.equiltime=equiltime
            self.state=state
            self.initsnap=self.state.get_snapshot()
            self.L=L
            self.N=N
            # self.gsd_file = gsd.hoomd.open(name=input, mode='w')

        def create_id(self):
            particleid=[]
            if self.initsnap.communicator.rank==0:
                positions=self.initsnap.particles.position
                particleid=self.initsnap.particles.typeid
                particleid[:1000]=particleid[:1000]*0
            # positions[1000]=[0,0,13]

                l=self.L
                N=len(positions[:1000])

                compartment=[[],[],[],[],[],[],[],[],[],[],[]]
                for i in range(N):
                    pos=positions[i][2]
                    # positions[i][2]=positions[i][2]+l/2
                    index=int(pos/2.5)

                    if index<=10:
                        compartment[int(index)].append(i)

                
                decay_const=2.59
                max_particles=54



                for num in range(11):
                    available = len(compartment[num])
                    if available == 0:
                        continue
                    parnum = int(np.exp(-num / decay_const) * max_particles)
                
                    take = min(parnum, available)
                    randomindex= np.random.choice(compartment[num], size=take, replace=False)
                    for index in randomindex:
                        particleid[index]=3
                for j in range(2):
                    particleid[j]=1
                    particleid[1000-(j+1)]=1

                    
            return particleid
        

        def create_snap(self):
            oldstate=self.state.get_snapshot()
            
            
            frame = gsd.hoomd.Frame()
            if oldstate.communicator.rank==0:
                while True:
                    ptypeid=self.create_id()
                    para=np.where(ptypeid==3)
                
                    if len(para[0])==ParAnum:
                        print(len(para[0]),'ready')
                        break
                    else:
                        continue
            
                frame.configuration.box = oldstate.configuration.box
                frame.particles.N = oldstate.particles.N
                frame.particles.types = oldstate.particles.types
                frame.particles.typeid = ptypeid
                frame.particles.position = oldstate.particles.position
                frame.bonds.N = oldstate.bonds.N
                frame.bonds.types = oldstate.bonds.types
                frame.bonds.typeid = oldstate.bonds.typeid
                frame.bonds.group = oldstate.bonds.group
            fsnapshot=hoomd.Snapshot.from_gsd_frame(frame,hoomd.communicator.Communicator())
            # self.gsd_file.append(frame)
            # self.gsd_file.close()
            return fsnapshot

        def act(self, timestep):
            fsnap= self.create_snap()
            self.state.set_snapshot(fsnap)

#----------------------------------Extrusion-----------------------------------------------------------------------------------------------------------------------
    class Extrusion(hoomd.custom.Action):
        def __init__(self,sim_object,custombond,state,addingrate,extrusion_rate,initparticle,equiltime):
            self.state=state
            self.sim=sim_object
            self.bond=custombond
            self.initparticle=initparticle
            self.isnap=self.state.get_snapshot()
            self.addmon=addingrate
            self.extru_rate=extrusion_rate
            self.i=0
            self.equiltime=equiltime
            self.box=self.isnap.configuration.box
            self.crosssize=0

        def ival(self,timestep):
            forkpos=int(timestep-self.equiltime)/self.addmon
            return forkpos

        def croslink(self,timestep):
                self.i=self.ival(timestep)-1
                
                numbond=0
                crossbonds=[]
                j=int((timestep-self.equiltime)/self.extru_rate)-1
                if self.sim.device.communicator.rank==0:
                            print(f'\rj={j} , timestep={timestep}',end='')
                            # print(f'j={j} , timestep={timestep}')
                if (timestep-self.equiltime)%self.extru_rate==0 and self.i>0 :
                    crossbonds=[]
                    j=int(int(timestep-self.equiltime)/self.extru_rate)-1
                    self.crosslinks=[[166,832],[333,666]]
                    for [a,b] in self.crosslinks:
                        minj=int((a-1)/1.25 +1)+1
                        if(a==166):
                            minj=10
                        
                            if j>=minj and j< minj +abs(a):
                                crossbonds.append([self.initparticle+2*(j-minj),self.initparticle+2*(j-minj)+1])
                                crossbonds.append([j-minj,(initparticles-1-(j-minj))])
                                numbond=numbond+2
                            if j>minj +abs(a):
                                crossbonds.append([self.initparticle+2*a,self.initparticle+2*a+1])
                                crossbonds.append([a,1009-a])
                                numbond=numbond+2
                        if(a==333):
                            minj=12
                        
                            if j>=minj and j< minj +abs(a):
                                crossbonds.append([self.initparticle+2*(j-minj),self.initparticle+2*(j-minj)+1])
                                crossbonds.append([j-minj,(initparticles-1-(j-minj))])
                                numbond=numbond+2
                            if j>minj +abs(a):
                                crossbonds.append([self.initparticle+2*a,self.initparticle+2*a+1])
                                crossbonds.append([a,1009-a])
                                numbond=numbond+2

                return np.array(crossbonds),numbond
        
        def create_new_snap1(self,timestep):
            self.isnap=self.state.get_snapshot()
            positions=[]
            ptypes=[]
            ptypeid=[]
            bgroup=[]
            btypes=[]
            btypeid=[]
            vel=[]
            if self.isnap.communicator.rank==0:
                positions=self.isnap.particles.position
                ptypeid = self.isnap.particles.typeid
                vel=self.isnap.particles.velocity
                ptypes=self.isnap.particles.types
                bgroup=self.isnap.bonds.group
                btypeid=self.isnap.bonds.typeid
                btypes=self.isnap.bonds.types
                crossbonds=[]
                self.crosslinks=[[166,832],[333,666]]
                maxj=int((333-1)/1.25 +1)+1 +abs(333)+2
                crossbonds,numbonds=self.croslink(timestep)
                j=int((timestep-self.equiltime)/self.extru_rate)-1
                if j<maxj and j>=0 and (timestep-self.equiltime)%self.extru_rate==0:
                    # bondtune=BondTuner(self.sim.operations.integrator.forces[2],0,100,1)
                    # self.sim.operations.updaters.remove(self.bond)
                    # self.bond=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(100),action=bondtune)
                    # self.sim.operations.updaters.append(self.bond)
                    j=int(int(timestep-self.equiltime)/self.extru_rate)-1
                    self.crosslinks=[[166,832],[333,666]]
                    count=len(self.crosslinks)-1
                    count=0
                    minj=[10,12]

                    if j in minj:
                        idxs=minj.index(j)
                        index = np.where((bgroup == self.crosslinks[idxs]).all(axis=1))[0][0]
                        bgroup = np.delete(bgroup, index, axis=0)
                        btypeid=np.delete(btypeid,index, axis=0)



                    minj=[12,10]
                    count=len(minj)-1

                    for minimj in minj[:]:
                        if minimj <= j:
                            if len(crossbonds) > 0:
                                if j==minimj:
                                    if len(bgroup) > 0:
                                        bgroup=np.vstack((crossbonds,bgroup[count*2:]))
                                        c=np.array([1,1])
                                        btypeid=np.append(c,btypeid)
                                        
                                if j>minimj:
                                    if len(bgroup) > 0:
                                        bgroup=np.vstack((crossbonds,bgroup[numbonds:]))

                        count=count-1
            frame=gsd.hoomd.Frame()
            frame = gsd.hoomd.Frame()
            frame.configuration.box = self.box
            frame.particles.N = len(positions)
            frame.particles.types = ptypes
            frame.particles.typeid = ptypeid
            frame.particles.position = positions
            frame.bonds.N = len(bgroup)
            frame.bonds.types = btypes
            frame.bonds.typeid = btypeid
            frame.bonds.group = bgroup
            frame.particles.velocity=vel
            fsnapshot=hoomd.Snapshot.from_gsd_frame(frame,hoomd.communicator.Communicator())
            return fsnapshot
        
        def act(self,timestep):
            fsnap=self.create_new_snap1(timestep)
            self.state.set_snapshot(fsnap)
            # top,bottom,dll=self.wall_shift(timestep)
            # self.wall.walls[0] = top
            # self.wall.walls[1] = bottom

#------------------------------------Main Program----------------------------------------------------------------------------------------------------------------------------

    def main(snapshot,addmon,L,A,equiltime,initparticles):
            # changing_initconfig()
            N=initparticles
            epsilon = 1
            initk=0
            finalk=100
            rate=1
            A=A
            L=L
            # comm = hoomd.communicator.Communicator(domain_decomposition=(1, 1, 4))
            # device = hoomd.device.CPU(communicator=comm)
            device = hoomd.device.CPU()
            # domain = hoomd.Domain(grid=(1,1,4))
            randmseed=np.random.randint(1,1000)
            sim = hoomd.Simulation(device=device, seed=randmseed)

            # setting harmonic potential
            harmonic = hoomd.md.bond.Harmonic()
            harmonic.params['A-A'] = dict(k=100, r0=1.0)
            harmonic.params['crosslink']=dict(k=100,r0=1.0)
            harmonic.params['teatheredbond']=dict(k=10,r0=1.0)
            
           
            # nlist=hoomd.md.nlist.NeighborList(buffer=10)
            # nlist = hoomd.md.nlist.Stencil(cell_width=10,buffer=0.8, rebuild_check_delay=1)
            # nlist=hoomd.md.nlist.Tree(buffer=0.8,rebuild_check_delay=1,check_dist=True)

            nlist=hoomd.md.nlist.Cell(buffer=0.8,rebuild_check_delay=1,check_dist=True)
            # setting LJ potential
            # sigma=0.6
            # lj=hoomd.md.pair.ForceShiftedLJ(nlist=nlist,default_r_cut=0)
            # types = ['A', 'ParB', 'teather','ParA', 'replisome', 'D2']

            # for i in range(len(types)):
            #     for j in range(i, len(types)):
            #         type_i = types[i]
            #         type_j = types[j]
            #         lj.params[(type_i, type_j)] = dict(epsilon=0, sigma=sigma)
            #         lj.r_cut[(type_i, type_j)]= 0
            # lj.params[('ParA','ParB')]=dict(epsilon=3,sigma=sigma)
            # lj.r_cut[('ParA','ParB')]=2


            attractivesoft=hoomd.md.pair.Buckingham(nlist,default_r_cut=0,mode='xplor')
            types = ['A', 'ParB', 'teather1','ParA', 'replisome', 'D2','teather2']

            for i in range(len(types)):
                for j in range(i, len(types)):
                    type_i = types[i]
                    type_j = types[j]
                    attractivesoft.params[(type_i, type_j)] = dict(A=0, rho=0.6, C=0)
                    attractivesoft.r_cut[(type_i, type_j)]= 0
            attractivesoft.params[('ParA','ParB')]=dict(A=-20, rho=0.6, C=0)
            attractivesoft.r_cut[('ParA','ParB')]=1.5  







            #Creating soft potential for replicating monomers
            
            soft=hoomd.md.pair.Buckingham(nlist,default_r_cut=0.8,mode='xplor')
            soft.params[('A', 'A')] = dict(A=A, rho=0.6, C=0)
            soft.params[('A', 'ParB')] = dict(A=A, rho=0.6, C=0)
            soft.params[('A', 'teather1')] = dict(A=0, rho=0.6, C=0)
            soft.params[('A', 'ParA')] = dict(A=A, rho=0.6, C=0)
            soft.params[('A', 'replisome')] = dict(A=A, rho=0.6, C=0)
            soft.params[('A', 'D2')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParB', 'ParB')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParB', 'replisome')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParB', 'ParA')] = dict(A=15, rho=0.2, C=0)
            # soft.params[('ParB', 'ParA')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParB', 'D2')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParB', 'teather1')] = dict(A=0, rho=0.6, C=0)
            soft.params[('teather1', 'teather1')] = dict(A=0, rho=0.6, C=0)
            soft.params[('teather1', 'ParA')] = dict(A=0, rho=0.6, C=0)
            soft.params[('teather1', 'replisome')] = dict(A=0, rho=0.6, C=0)
            soft.params[('teather1', 'D2')] = dict(A=0, rho=0.6, C=0)
            soft.params[('ParA', 'ParA')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParA', 'replisome')] = dict(A=A, rho=0.6, C=0)
            soft.params[('ParA', 'D2')] = dict(A=A, rho=0.6, C=0)
            soft.params[('replisome', 'D2')] = dict(A=A, rho=0.6, C=0)
            soft.params[('replisome', 'replisome')] = dict(A=0, rho=0.6, C=0)
            soft.params[('D2','D2')] = dict(A=A, rho=0.6, C=0)
            soft.params[('teather2','teather2')]=dict(A=0, rho=0.6, C=0)
            soft.params[('teather1', 'teather2')] = dict(A=0, rho=0.6, C=0)
            soft.params[('teather2','A')]=dict(A=0, rho=0.6, C=0)
            soft.params[('teather2','ParA')]=dict(A=0, rho=0.6, C=0)
            soft.params[('teather2','ParB')]=dict(A=-30, rho=0.6, C=0)
            soft.params[('teather2','D2')]=dict(A=0, rho=0.6, C=0)
            soft.params[('teather2','replisome')]=dict(A=0, rho=0.6, C=0)
            soft.r_cut[('ParA','ParB')]=1.5
            soft.r_cut[('teather2','ParB')]=1.5

            # creating hard box
            z=L/2
            # z=10
            halfsigma=0.4
            top =  hoomd.wall.Plane(origin=(0, 0, 2*z), normal=(0, 0, -1))
            bottom = hoomd.wall.Plane(origin=(0, 0, 0), normal=(0, 0, 1))
            left = hoomd.wall.Plane(origin=(-z, 0, 0), normal=(1, 0, 0))
            right = hoomd.wall.Plane(origin=(z, 0, 0), normal=(-1, 0, 0))
            front = hoomd.wall.Plane(origin=(0, z, 0), normal=(0, -1, 0))
            back = hoomd.wall.Plane(origin=(0, -z, 0), normal=(0, 1, 0))
            ljbox = hoomd.md.external.wall.ForceShiftedLJ([top, bottom])#, left, right, back, front])
            ljbox.params['A'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['ParB'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['teather1'] = dict(epsilon=0, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['ParA'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['replisome'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['D2'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljbox.params['teather2'] = dict(epsilon=0, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)

            #Creating cylindrical confinement
            cylinder=hoomd.wall.Cylinder(radius=4,axis=(0,0,1))
            ljcylinder=hoomd.md.external.wall.ForceShiftedLJ([cylinder])
            ljcylinder.params['A'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['ParB'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['teather1'] = dict(epsilon=0, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['ParA'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['replisome'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['D2'] = dict(epsilon=1, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)
            ljcylinder.params['teather2'] = dict(epsilon=0, sigma=halfsigma, r_cut=(2.0 ** (1 / 6)) * halfsigma)


            # setting langevin

            lang = hoomd.md.methods.Langevin(filter=hoomd.filter.Type(['A','ParB','ParA','replisome','D2']), kT=1.0)
            
            #setting bondtuner
            bondtuner=BondTuner(harmonic,initk,finalk,rate)

            #setting soft tuner:
            softtune=softtuner(soft,0,A,0.5)
            integrator = hoomd.md.Integrator(dt=0.01, methods=[lang])

            
            integrator.forces.append(ljcylinder)
            integrator.forces.append(ljbox)
            integrator.forces.append(harmonic)
            integrator.forces.append(soft)
            # integrator.forces.append(attractivesoft)
            sim.operations.integrator = integrator
            sim.create_state_from_gsd(snapshot)
            # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
            # sim.run(0)
            sim.operations.integrator = integrator

            # writer1=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(5000),action=CustomWriteFrames(output))
            try:
                writer1=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(5000),action=CustomWriteFrames(output))
            except Exception as e:
                print(f"Error writing to file: {e}")
                time.sleep(1)  # wait for 1 second before trying again
                writer1=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(5000),action=CustomWriteFrames(output))
            custombond=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(100),action=bondtuner)
            customsoft=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(50),action=softtune)
            customreplication=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(int(addmon)) 
                                                         ,action=Replication(customsoft,custombond,sim,sim.state,ljbox,addmon,addmon*1.25
                                                                             ,initparticles,equiltime,L))
            customextrusion=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(int(addmon*2)) 
                                                         ,action=Extrusion(sim,custombond,sim.state,addmon,int(addmon*2),initparticles,equiltime))
            
            customParA=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(ParArate),action=ParAredist(sim,sim.state,equiltime,nlist,addmon,L,initparticles))
            
            custominit=hoomd.update.CustomUpdater(trigger=hoomd.trigger.On(int(equiltime-1)),action=initmod(sim.state,equiltime,L,N,None))
            sim.operations.updaters.append(custominit)
            # sim.operations.writers.append(writer1)
            if device.communicator.rank==0:
                print(f'\rRun Start for equillibration {iter}',end='\n')
            sim.run(equiltime)
            if device.communicator.rank==0:
                print(f'\rRun end for equillibration {iter}',end='\n')
            
            sim.operations.integrator.forces.append(attractivesoft)
            # sim.run(0)
            # writer2=hoomd.write.gsd.GSD(trigger=hoomd.trigger.Periodic(100),filename='checktraj.gsd',mode='ab')
            sim.operations.updaters.append(custombond)
            sim.operations.writers.append(writer1)
            # sim.operations.writers.append(writer2)
            sim.operations.updaters.append(customsoft)
            sim.operations.updaters.append(customextrusion)
            sim.operations.updaters.append(customParA)
            sim.operations.updaters.append(customreplication)
            
            
            if device.communicator.rank==0:
                print(f'\rRun Start for iteration {iter}',end='\n')
            try:
                # sim.run(40000)
                # sim.run(int((addmon*initparticles*6/10)))#+1e6))

                sim.run(int((addmon*initparticles*2/10)))#+1e6))
                # sim.operations.updaters.append(customParA)
                # sim.run(int((addmon*initparticles*1/10)))
            finally:
                writer1.action.close()

                sim.operations.writers.remove(writer1)
                # writer1.action.close()
            
            if device.communicator.rank==0:
                print(f"\rRun end for iteration {iter}",end='\n')
            
            
    main(inputfile,addmon,L,A,equiltime,initparticles)
    end_time=time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
