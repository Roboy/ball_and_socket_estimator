import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math
from multiprocessing import Pool, freeze_support, get_context, set_start_method
from yaml import load,dump,Loader,Dumper
import sys, time
import rospy
from tqdm import tqdm
import pcl
import pcl.pcl_visualization

class BallJoint:
    config = {}
    sensors = None
    sampling_method = 'grid'
    min_rot_x,min_rot_y,min_rot_z = -50,-50,-50
    max_rot_x,max_rot_y,max_rot_z = 50,50,50
    grid_positions = [[0,0,0]]
    normalize_magnetic_field = False
    num_processes = 60
    sensor_values = []
    pos_values = []
    number_of_samples = 0
    magnetic_field_difference_sensitivity = 1.44

    def __init__(self,config_file_path):
        self.config = load(open(config_file_path, 'r'), Loader=Loader)
        self.printConfig()
        self.sensors = self.gen_sensors()

    def visualizeCloud(self,magnitudes,pos_values):
        cloud = pcl.PointCloud_PointXYZRGB()
        points = np.zeros((len(magnitudes), 4), dtype=np.float32)
        i = 0
        for rot in pos_values:
            points[i][0] = magnitudes[i]*math.sin(rot[0]/180.0*math.pi)*math.cos(rot[1]/180.0*math.pi)
            points[i][1] = magnitudes[i]*math.sin(rot[0]/180.0*math.pi)*math.sin(rot[1]/180.0*math.pi)
            points[i][2] = magnitudes[i]*math.cos(rot[0]/180.0*math.pi)
            points[i][3] = 255 << 16 | 255 << 8 | 255
            i = i+1

        cloud.from_array(points)

        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorCloud(cloud)

        v = True
        while v:
            v = not(visual.WasStopped())

    def rotateMagnets(self,magnets,rot):
        magnets.rotate(rot[0],(1,0,0), anchor=(0,0,0))
        magnets.rotate(rot[1],(0,1,0), anchor=(0,0,0))
        magnets.rotate(rot[2],(0,0,1), anchor=(0,0,0))
        return magnets

    def collisionFunc(self,i):
        collision_indices = []
        magnetic_field_difference = 0
        magnetic_field_differences = []
        # position_difference = []
        global iterations
        collisions = 0
        for j in range(i+1,self.number_of_samples):
            mag_diff = 0
            for k in range(0,4):
                mag_diff += np.linalg.norm(self.sensor_values[i][k]-self.sensor_values[j][k])
            magnetic_field_difference+=mag_diff
            # pos_diff = np.linalg.norm(pos[i]-pos[j])
            # position_difference.append(pos_diff)
            if mag_diff<self.magnetic_field_difference_sensitivity:# and pos_diff>position_difference_sensitivity:
                collisions+=1
                collision_indices.append([i,j])
                magnetic_field_differences.append(mag_diff)
                # if(collisions%100==0):
                #     print("collisions: %d"%collisions)

        return (collisions,collision_indices,magnetic_field_difference,magnetic_field_differences)
    def calculateCollisions(self,sensor_values,pos_values,magnetic_field_difference_sensitivity):
        self.magnetic_field_difference_sensitivity = magnetic_field_difference_sensitivity
        self.number_of_samples = len(sensor_values)
        self.sensor_values = sensor_values
        self.pos_values = pos_values
        comparisons = (self.number_of_samples-1)*self.number_of_samples/2
        print('\ncomparisons %d'%comparisons)
        print('approx time %d seconds or %f minutes'%(comparisons/1283370,comparisons/1283370/60))
        timestamp = time.strftime("%H:%M:%S")
        print('start time: %s'%timestamp)
        args = range(0,self.number_of_samples,1)
        with Pool(processes=self.num_processes) as pool:
            start = time.time()
            results = pool.starmap(self.collisionFunc, zip(args))
            collisions = 0
            magnetic_field_difference = 0
            colliders = []
            magnetic_field_differences = []
            for n in range(0,self.number_of_samples):
                collisions += results[n][0]
                magnetic_field_difference += results[n][2]
                i = 0
                for indices in results[n][1]:
                    colliders.append([pos_values[indices[0]],pos_values[indices[1]]])
                    magnetic_field_differences.append(results[n][3][i])
        end = time.time()
        print('actual time: %d'%(end - start))
        print('\ncollisions: %d'%collisions)
        print('average magnetic_field_difference: %f\n'%(magnetic_field_difference/comparisons))
        return colliders,magnetic_field_differences
    def generateMagneticDataRandom(self,number_of_samples):
        self.sampling_method = 'random'
        args = range(0,number_of_samples,1)
        with Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self.generateMagneticData, zip(args))
        return results[0],results[1]

    def generateMagneticDataGrid(self,grid_positions):
        print('generating %d grid_positions'%len(grid_positions))
        start = time.time()
        self.grid_positions = grid_positions
        self.sampling_method = 'grid'
        args = range(0,len(self.grid_positions),1)
        with Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self.generateMagneticData, zip(args))
        sensor_values = []
        pos = []
        for res in results:
            sensor_values.append(res[0])
            pos.append(res[1])
        end = time.time()
        print('took: %d s or %f min'%(end - start,(end - start)/60))
        return sensor_values,pos

    def writeMagneticData(self,sensor_values,positions,filename):
        record = open('data/'+filename,"w")
        record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")
        status = tqdm(total=len(sensor_values), desc='write_to_file', position=0)
        for sens,pos in zip(sensor_values,positions):
            record.write(\
                str(sensor_values[0][0])+ " " + str(sensor_values[0][1]) + " " + str(sensor_values[0][2])+ " " + \
                str(sensor_values[1][0])+ " " + str(sensor_values[1][1])+ " " + str(sensor_values[1][2])+ " " + \
                str(sensor_values[2][0])+ " " + str(sensor_values[2][1])+ " " + str(sensor_values[2][2])  + " " + \
                str(sensor_values[3][0])+ " " + str(sensor_values[3][1])+ " " + str(sensor_values[3][2])+ " " + \
                str(pos[0]/180.0*math.pi) + " " + str(pos[1]/180.0*math.pi) + " " + str(pos[2]/180.0*math.pi) + "\n")
            status.update(1)
        record.close()
    def generateMagneticData(self,iter):
        if self.sampling_method=='random':
            rot = [random.uniform(min_rot_x,max_rot_x),\
                    random.uniform(min_rot_y,max_rot_y),\
                    random.uniform(min_rot_z,max_rot_z)]
        elif self.sampling_method=='grid':
            rot = self.grid_positions[iter]

        magnets = self.gen_magnets()
        magnets.rotate(rot[0],(1,0,0), anchor=(0,0,0))
        magnets.rotate(rot[1],(0,1,0), anchor=(0,0,0))
        magnets.rotate(rot[2],(0,0,1), anchor=(0,0,0))
        data = []
        for sens in self.sensors:
            val = sens.getB(magnets)
            if self.normalize_magnetic_field:
                val /= np.linalg.norm(val)
            data.append(val)
        return data, rot

    def gen_sensors(self):
        sensors = []
        for pos,pos_offset,angle,angle_offset in zip(self.config['sensor_pos'],\
            self.config['sensor_pos_offsets'],self.config['sensor_angle'],self.config['sensor_angle_offsets']):
            s = Sensor(pos=(pos[0]+pos_offset[0],pos[1]+pos_offset[1],pos[2]+pos_offset[2]))
            s.rotate(angle=angle[0]+angle_offset[0],axis=(1,0,0))
            s.rotate(angle=angle[1]+angle_offset[1],axis=(0,1,0))
            s.rotate(angle=angle[2]+angle_offset[2],axis=(0,0,1))
            sensors.append(s)
        return sensors

    def gen_magnets(self):
        magnets = []
        for field,mag_dim,pos,pos_offset,angle,angle_offset in zip(self.config['field_strength'],\
            self.config['magnet_dimension'],self.config['magnet_pos'],self.config['magnet_pos_offsets'],\
            self.config['magnet_angle'],self.config['magnet_angle_offsets']):
            magnet = Box(mag=(0,0,field), dim=mag_dim,\
                pos=(pos[0]+pos_offset[0],\
                    pos[1]+pos_offset[1],\
                    pos[2]+pos_offset[2])\
                    )
            magnet.rotate(angle=angle[0]+angle_offset[0],axis=(1,0,0))
            magnet.rotate(angle=angle[1]+angle_offset[1],axis=(0,1,0))
            magnet.rotate(angle=angle[2]+angle_offset[2],axis=(0,0,1))
            magnets.append(magnet)
        return Collection(magnets)

    def plotMagnets(self,magnets):
        # calculate B-field on a grid
        xs = np.linspace(-25,25,50)
        ys = np.linspace(-25,25,50)
        zs = np.linspace(-25,25,50)
        POS0 = np.array([(x,0,z) for z in zs for x in xs])
        POS1 = np.array([(x,y,0) for y in ys for x in xs])

        fig = plt.figure(figsize=(18,7))
        ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(133)                   # 2D-axis
        ax3 = fig.add_subplot(132)                   # 2D-axis

        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        Bs = magnets.getB(POS0).reshape(50,50,3)     #<--VECTORIZED
        X,Y = np.meshgrid(xs,zs)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        Bs = magnets.getB(POS1).reshape(50,50,3)     #<--VECTORIZED
        X,Z = np.meshgrid(xs,ys)
        U,V = Bs[:,:,0], Bs[:,:,1]
        ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
        displaySystem(magnets, subplotAx=ax1, suppress=True, sensors=self.sensors, direc=True)
        plt.show()
    def compareMagnets(self,magnet_A,magnet_B):
        # calculate B-field on a grid
        xs = np.linspace(-25,25,50)
        ys = np.linspace(-25,25,50)
        zs = np.linspace(-25,25,50)
        POS0 = np.array([(x,0,z) for z in zs for x in xs])
        POS1 = np.array([(x,y,0) for y in ys for x in xs])

        fig = plt.figure(figsize=(18,14))
        ax1 = fig.add_subplot(231, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(233)                   # 2D-axis
        ax3 = fig.add_subplot(232)                   # 2D-axis
        ax4 = fig.add_subplot(234, projection='3d')  # 3D-axis
        ax5 = fig.add_subplot(236)                   # 2D-axis
        ax6 = fig.add_subplot(235)                   # 2D-axis

        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        Bs = magnet_A.getB(POS0).reshape(50,50,3)     #<--VECTORIZED
        XA,YA = np.meshgrid(xs,zs)
        UA,VA = Bs[:,:,0], Bs[:,:,2]
        ax2.streamplot(XA, YA, UA, VA, color=np.log(UA**2+VA**2))

        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        Bs = magnet_A.getB(POS1).reshape(50,50,3)     #<--VECTORIZED
        XA,ZA = np.meshgrid(xs,ys)
        UA,VA = Bs[:,:,0], Bs[:,:,1]
        ax3.streamplot(XA, ZA, UA, VA, color=np.log(UA**2+VA**2))

        ax5.set_xlabel('x')
        ax5.set_ylabel('z')
        Bs = magnet_B.getB(POS0).reshape(50,50,3)     #<--VECTORIZED
        XB,YB = np.meshgrid(xs,zs)
        UB,VB = Bs[:,:,0], Bs[:,:,2]
        ax5.streamplot(XB, YB, UB, VB, color=np.log(UB**2+VB**2))

        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        Bs = magnet_B.getB(POS1).reshape(50,50,3)     #<--VECTORIZED
        XB,ZB = np.meshgrid(xs,ys)
        UB,VB = Bs[:,:,0], Bs[:,:,1]
        ax6.streamplot(XB, ZB, UB, VB, color=np.log(UB**2+VB**2))

        displaySystem(magnet_A, subplotAx=ax1, suppress=True, sensors=self.sensors, direc=True)
        displaySystem(magnet_B, subplotAx=ax4, suppress=True, sensors=self.sensors, direc=True)
        plt.show()

    def printConfig(self):
        print("id: %d"%self.config['id'])
        print("calibration")
        self.config['calibration']
        print("field_strength")
        print(self.config['field_strength'])
        print("magnet_pos")
        for val in self.config['magnet_pos']:
            print(val)
        print("magnet_pos_offsets")
        for offset in self.config['magnet_pos_offsets']:
            print(offset)
        print("magnet_angle")
        for val in self.config['magnet_angle']:
            print(val)
        print("magnet_angle_offsets")
        for offset in self.config['magnet_angle_offsets']:
            print(offset)
        print("sensor_pos")
        for val in self.config['sensor_pos']:
            print(val)
        print("sensor_pos_offsets")
        for offset in self.config['sensor_pos_offsets']:
            print(offset)
        print("sensor_angle")
        for val in self.config['sensor_angle']:
            print(val)
        print("sensor_angle_offsets")
        for offset in self.config['sensor_angle_offsets']:
            print(offset)
