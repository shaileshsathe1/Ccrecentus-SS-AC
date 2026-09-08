import hoomd
import gsd.hoomd
import numpy as np
import time
import hoomd.azplugins
import os
import h5py
import time

start_time = time.time()
monnum=[1000]
 



#---------------------framewriter--------------------------------
class writingframes(hoomd.custom.Action):
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
            print(f"\rFrame {self.frames_written} written at timestep {timestep}",end='')





#--------------------Softtuner------------------------------------
class softtuner(hoomd.custom.Action):
        def __init__(self,pair_force,initA,finalA,rate):
            self.pair_force=pair_force
            self.k=initA
            self.finalK=finalA
            self.rate=rate
        def act(self, timestep):
            if self.k < self.finalK:
                self.k += self.rate
                self.pair_force.params[('A', 'A')] = dict(A=self.k, rho=0.6 , C=0)


#--------------------bondtuner------------------------------------
class BondTuner(hoomd.custom.Action):
        def __init__(self,bond_force,initK,finalK,rate):
            self.bond_force=bond_force
            self.k=initK
            self.finalK=finalK
            self.rate=rate
        def act(self, timestep):
            if self.k < self.finalK:
                self.k += self.rate
                self.bond_force.params['crosslink'] = dict(k=self.k, r0=1)

#--------------------------Harmonic cylinder-------------------------
class HarmonicCylinder(hoomd.md.force.Custom):
    def __init__(self, radius, k, axis=(0, 0, 1)):
        # Initialize the CustomForceCompute with the provided trigger
        super().__init__()
        self.radius = radius
        self.k = k
        self.axis = np.array(axis) / np.linalg.norm(axis)

    def set_forces(self, timestep):
        # print(timestep)
        i=0
        with self.cpu_local_force_arrays as arrays, self._state.cpu_local_snapshot as snap:
            forces = np.zeros_like(snap.particles.position)
            positions = snap.particles.position
            nparticles=len(positions)
            for  i in range(nparticles):
                force_array = arrays.force
                radial_pos=np.zeros(3)
                radial_pos[0:2] = positions[i][0:2]  # Project onto plane perpendicular to axis
                radial_dist = np.linalg.norm(radial_pos)
                # print(radial_dist,positions[i])
                # print(pos)
                if radial_dist > self.radius:
                    # print(i) 
                    displacement = radial_dist - self.radius
                    radial_dir = radial_pos / radial_dist
                    force_magnitude =   - self.k * displacement
                    forces[i] = force_magnitude * radial_dir
                    force_array[i] = forces[i]

#--------------------maincode---------------------------------------

def main(output,num):
    N =500
    sigma = 0.7
    KT = 1.0
    A=5
    runs=5e4
    L=28
    DD = hoomd.device.auto_select()
    randmseed=np.random.randint(1,1000)
    sim = hoomd.Simulation(device=DD, seed=randmseed)

    # setting harmonic potential
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['A-A'] = dict(k=100, r0=1.0)
    harmonic.params['crosslink']=dict(k=100,r0=1.0)
    harmonic.params['teatheredbond']=dict(k=10,r0=1.0)
    

    # setting LJ potential
    
    # nlist = hoomd.md.nlist.Stencil(cell_width=10,buffer=0.8, rebuild_check_delay=1, check_dist=False,default_r_cut=2)
    
    nlist=hoomd.md.nlist.Cell(buffer=0.8,rebuild_check_delay=1,check_dist=True)


    
    #Creating soft potential for replicating monomers
    soft=hoomd.md.pair.Buckingham(nlist,default_r_cut=0.8,mode="xplor")
    soft.params[('A', 'A')] = dict(A=A, rho=0.6, C=0)
    soft.params[('A', 'ParB')] = dict(A=A, rho=0.6, C=0)
    soft.params[('A', 'teather1')] = dict(A=0, rho=0.6, C=0)
    soft.params[('A', 'ParA')] = dict(A=A, rho=0.6, C=0)
    soft.params[('A', 'replisome')] = dict(A=A, rho=0.6, C=0)
    soft.params[('A', 'D2')] = dict(A=A, rho=0.6, C=0)
    soft.params[('ParB', 'ParB')] = dict(A=A, rho=0.6, C=0)
    soft.params[('ParB', 'replisome')] = dict(A=A, rho=0.6, C=0)
    soft.params[('ParB', 'ParA')] = dict(A=5, rho=0.6, C=0)
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
    soft.r_cut[('ParA','ParB')]=1
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
    # harmonicwalls=hoomd.azplugins.external.HarmonicBarrier(cylinder)
    # harmonicwalls.params['A']=dict(k=10,offset=0.5)
    # harmonicwalls.params['ParB']=dict(k=10,offset=0.5)
    # harmonicwalls.params['teather']=dict(k=10,offset=0.5)
    # setting langevin

    # cylinder=HarmonicCylinder(radius=5,k=100)
    integrator = hoomd.md.Integrator(dt=0.01, methods=[lang])
    integrator.forces.append(ljcylinder)
    integrator.forces.append(ljbox)
    integrator.forces.append(harmonic)
    integrator.forces.append(soft)
    # integrator.forces.append(cylinder)
    sim.operations.integrator = integrator

    sim.create_state_from_gsd(f'initfiles/equiled_1000_28_4_noncrosslinked.gsd')
    # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
    # sim.run(0)
    # thermoprop=hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    # sim.operations.computes.append(thermoprop)
    # logger=hoomd.logging.Logger(hoomd.write.HDF5Log.accepted_categories)
    # logger.add(thermoprop)
    # logger.add(sim,quantities=['timestep','walltime'])
    # propwriter=hoomd.write.HDF5Log(trigger=hoomd.trigger.Periodic(5000),filename='outputtrajs/4000mddata2.h5',mode='w',logger=logger)
    # sim.operations.writers.append(propwriter) 
    sim.operations.integrator = integrator
    # softtune=softtuner(soft,0,A,0.01)
    bondtune=BondTuner(harmonic,0,100,0.01)
    # costumsoft=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(10),action=softtune)
    costumbond=hoomd.update.CustomUpdater(trigger=hoomd.trigger.Periodic(10),action=bondtune)
    
    sim.operations.updaters.append(costumbond)
    # sim.operations.updaters.append(costumsoft)
    
    # writer2 = hoomd.write.GSD(filename=f'outputtrajs/{num}_polymertrajectory.gsd', trigger=hoomd.trigger.Periodic(100), mode='wb')
    # neighbourlist=hoomd.md.nlist.NeighborList.pair_list()
    # print(neighbourlist)
    try:
        writer1=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(5000),action=writingframes(output))
    except Exception as e:
        print(f"Error writing to file: {e}")
        time.sleep(1)  # wait for 1 second before trying again
        writer1=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(5000),action=writingframes(output))
    sim.operations.writers.append(writer1)
    # sim.operations.writers.append(writer2)
    if DD.communicator.rank==0:

        print('Run Start')
    
    start_time = time.time()
    sim.run(1e4)

    # step=hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(1000),action=StepWriter())
    # sim.operations.writers.append(step)
    # sim.operations.updaters.remove(costumbond)
    # sim.operations.updaters.remove(costumsoft)
    
    sim.run(1e8)
    
    end_time = time.time()
    if DD.communicator.rank==0:
        print(f"Run end time taken to complete run={end_time-start_time} seconds")
    fsnapshot=sim.state.get_snapshot()
    if fsnapshot.communicator.rank == 0:
        frame = gsd.hoomd.Frame()
        frame.configuration.box = fsnapshot.configuration.box
        frame.particles.N = fsnapshot.particles.N
        frame.particles.types = fsnapshot.particles.types
        frame.particles.typeid = fsnapshot.particles.typeid
        frame.particles.position = fsnapshot.particles.position
        frame.bonds.N = fsnapshot.bonds.N
        frame.bonds.types = fsnapshot.bonds.types
        frame.bonds.typeid = [0]*len(fsnapshot.bonds.group)
        frame.bonds.group = fsnapshot.bonds.group
        
        equiledfile = gsd.hoomd.open(f'initfiles/equiled_1000_28_4_noncrosslinked1.gsd', mode='w')
        equiledfile.append(frame)
        equiledfile.close()
        print("Snapshot successfully written to the GSD file.")
    # writer2.flush()

for num in monnum:
     for k in range(1):
        A=5
        os.makedirs('outputtrajs',exist_ok=True)
        output=f'outputtrajs/loci_traj_{A}_{num}1.gsd'
        main(output,num)