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
from tqdm import tqdm
import pcl
import pcl.pcl_visualization
import rospy
from roboy_middleware_msgs.msg import MagneticSensor

class BallJoint:
    config = {}
    config_file = None
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
    number_of_sensors = 0
    number_of_magnets = 0
    number_of_parameters = 0
    positions = []
    type = 'angle'
    calib = []
    distance = None

    def __init__(self,config_file_path):
        self.config = load(open(config_file_path, 'r'), Loader=Loader)
        self.printConfig()
        self.config_file = config_file_path
        self.number_of_sensors = len(self.config['sensor_pos'])
        self.number_of_magnets = len(self.config['field_strength'])
        self.sensors = self.gen_sensors()
    def generateSensorDataFork(self,i,joint_positions,magnet_angles):
        magnets = self.gen_magnets_angle(magnet_angles)
        magnets.rotate(joint_positions[i][0],(1,0,0), anchor=(0,0,0))
        magnets.rotate(joint_positions[i][1],(0,1,0), anchor=(0,0,0))
        magnets.rotate(joint_positions[i][2],(0,0,1), anchor=(0,0,0))
        values = []
        for sens in self.sensors:
            values.append(sens.getB(magnets))
            # displaySystem(magnets, suppress=False, sensors=self.sensors, direc=True)
        return values
    def generateSensorData(self,joint_positions,magnet_angles):
        # print('generating sensor data')
        number_of_positions = len(joint_positions)
        sensor_values = []
        with Pool() as pool:
            result = pool.starmap(self.generateSensorDataFork, zip(range(0,number_of_positions),\
                [joint_positions]*number_of_positions,[magnet_angles]*number_of_positions))
            return result
    def decodeX(self,x,type):
        positions = []
        angles = []
        if type=='posangle':
            for j in range(0,len(x),3):
                if j<len(x)/2:
                    positions.append([x[j],x[j+1],x[j+2]])
                else:
                    angles.append([x[j],x[j+1],x[j+2]])
        if type=='angle':
            for j in range(0,len(x),3):
                angles.append([x[j],x[j+1],x[j+2]])
            positions = self.positions
        return positions,angles

    def optimizeMagnetArrangement(self,type):
        positions = []
        angles = []
        for i in np.arange(0,360,30):
            for j in np.arange(0,360,30):
                positions.append([\
                    25*math.sin(i/180.0*math.pi)*math.cos(j/180.0*math.pi),\
                    25*math.sin(i/180.0*math.pi)*math.sin(j/180.0*math.pi),\
                    25*math.cos(i/180.0*math.pi)\
                ])
                angles.append([0,0,90])
        self.sensors = self.gen_sensors_custom(positions,angles)
        # magnets = self.gen_magnets()
        # print('number of sensors %d'%len(positions))
        # displaySystem(magnets, suppress=False, sensors=self.sensors, direc=True)
        positions = []
        if type=='posangle':
            x_init = [0]*self.number_of_magnets*3*2
            x_lower_bound = [0]*self.number_of_magnets*3*2
            x_upper_bound = [0]*self.number_of_magnets*3*2
            for i in range(0,self.number_of_magnets*3):
                x_init[i] = random.uniform(-12,12)
                x_lower_bound[i] = -12
                x_upper_bound[i] = 12
            for i in range(self.number_of_magnets*3,self.number_of_magnets*3*2):
                x_init[i] = random.uniform(-90,90)
                x_lower_bound[i] = -90
                x_upper_bound[i] = 90
        elif type=='angle':
            self.config['field_strength']=[]
            self.config['magnet_dimension'] = []
            for i in range(0,360,100):
                for j in range(60,300,100):
                    positions.append([\
                        10*math.sin(i/180.0*math.pi)*math.cos(j/180.0*math.pi),\
                        10*math.sin(i/180.0*math.pi)*math.sin(j/180.0*math.pi),\
                        10*math.cos(i/180.0*math.pi)\
                    ])
                    self.config['field_strength'].append(1300)
                    self.config['magnet_dimension'].append([7,7,7])
            self.number_of_magnets=len(positions)
            x_init = [0]*self.number_of_magnets*3
            x_lower_bound = [0]*self.number_of_magnets*3
            x_upper_bound = [0]*self.number_of_magnets*3
            for i in range(0,self.number_of_magnets*3):
                x_init[i] = 0#random.uniform(-90,90)
                x_lower_bound[i] = -90
                x_upper_bound[i] = 90
            self.positions = positions
        print("number_of_magnets: %d"%self.number_of_magnets)
        positions,angles = self.decodeX(x_init,type)
        magnets = self.gen_magnets_custom(positions,angles)
        displaySystem(magnets, suppress=False,direc=True)
        self.type = type
        def optimizeFun(x):
            positions,angles = self.decodeX(x,self.type)
            magnets = self.gen_magnets_custom(positions,angles)
            sensor_values = []
            for sens in self.sensors:
                sensor_values.append(sens.getB(magnets))
            b_error = 0
            for i in range(0,len(sensor_values)):
                for j in range(i+1,len(sensor_values)):
                    norm = np.linalg.norm(sensor_values[i]-sensor_values[j])
                    if(norm>0.001):
                        b_error += -math.log(norm)
                    else:
                        b_error += 1000
            return [b_error]
        res = least_squares(optimizeFun, x_init,\
            bounds = (x_lower_bound, x_upper_bound),\
            ftol=1e-8, \
            xtol=1e-8,verbose=2,\
            max_nfev=self.config['calibration']['max_nfev'])
        print(res)
        positions,angles = self.decodeX(res.x,type)
        magnets = self.gen_magnets_custom(positions,angles)
        displaySystem(magnets, suppress=False,direc=True)

    def decodeCalibrationX(self,x):
        field_strength = []
        magnet_pos_offsets = []
        magnet_angle_offsets = []
        sensor_pos_offsets = []
        sensor_angle_offsets = []

        for i in range(0,self.number_of_magnets):
            field_strength.append(1300)
            magnet_pos_offsets.append([0,0,0])
            magnet_angle_offsets.append([0,0,0])
        for i in range(0,self.number_of_sensors):
            sensor_pos_offsets.append([0,0,0])
            sensor_angle_offsets.append([0,0,0])
        j = 0
        for c in self.calib:
            if c==0:
                for i in range(0,self.number_of_magnets):
                    field_strength[i] = x[j]
                    j = j+1
            if c==1:
                for i in range(0,self.number_of_magnets):
                    magnet_pos_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==2:
                for i in range(0,self.number_of_magnets):
                    magnet_angle_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==3:
                for i in range(0,self.number_of_sensors):
                    sensor_pos_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==4:
                for i in range(0,self.number_of_sensors):
                    sensor_angle_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
        return field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets

    def calibrationFunc(self, x):
        sensor_pos = self.config['sensor_pos']
        sensor_angle = self.config['sensor_angle']

        field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets = self.decodeCalibrationX(x)

        sensors = self.gen_sensors_all(sensor_pos,sensor_pos_offsets,sensor_angle,sensor_angle_offsets)

        b_error = 0
        j =0
        for pos,angle in zip(self.config['calibration']['magnet_pos'],self.config['calibration']['magnet_angle']):
            magnets = self.gen_magnets_all(field_strength,[pos],magnet_pos_offsets,[angle],magnet_angle_offsets)
            # print("%f %f %f"%(pos[0],pos[1],pos[2]))
            i = 0
            for sens in sensors:
                b_error = b_error + np.linalg.norm(sens.getB(magnets)-self.sensor_values[j][i])
                i=i+1
            j =j +1
        # print(b_error)
        return [b_error]

    def calibrateSensor(self):
        rospy.init_node('BallJoint',anonymous=True)
        print('calibrating sensor')
        print('calibration magnet positions')
        print(self.config['calibration']['magnet_pos'])
        print('calibration magnet angles')
        print(self.config['calibration']['magnet_angle'])
        calibration_status = tqdm(total=len(self.config['calibration']['magnet_pos']), desc='calibration_status', position=0)
        self.sensor_values = []
        sensor_log = {'magnet_pos':[],'magnet_angle':[],'sensor_values':[]}
        for pos,angle in zip(self.config['calibration']['magnet_pos'],self.config['calibration']['magnet_angle']):
            print(pos)
            print(angle)
            sensor_log['magnet_pos'].append(pos)
            sensor_log['magnet_angle'].append(angle)
            magnets = self.gen_magnets_all(self.config['field_strength'],[pos],self.config['magnet_pos_offsets'],[angle],self.config['magnet_angle_offsets'])
            print('target:')
            for sens in self.sensors:
                print(sens.getB(magnets))
            displaySystem(magnets, suppress=False, sensors=self.sensors, direc=True)

            values = []
            for i in range(0,self.number_of_sensors):
                values.append([0,0,0])
            sample = 0
            sensor_record_status = tqdm(total=100, desc='sensor_record_status', position=1)
            while sample<100:
                msg = rospy.wait_for_message("/roboy/middleware/MagneticSensor", MagneticSensor, timeout=None)
                if(msg.id==self.config['id']):
                    j = 0
                    for x,y,z in zip(msg.x,msg.y,msg.z):
                        values[j][0]+=x
                        values[j][1]+=y
                        values[j][2]+=z
                        j+=1
                    sample+=1
                    sensor_record_status.update(1)
            for j in range(0,self.number_of_sensors):
                values[j][0]/=sample
                values[j][1]/=sample
                values[j][2]/=sample

            self.sensor_values.append(values)
            sensor_log['sensor_values'].append(values)
            calibration_status.update(1)
        print("optimizing: ")
        initial = []
        upper_bound = []
        lower_bound = []
        for c in self.config['calibration']['optimize']:
            if c=='field_strength':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets
                for i in range(0,self.number_of_magnets):
                    initial.append(1300)
                    upper_bound.append(1600)
                    lower_bound.append(1000)
                self.calib.append(0)
                print('\tfield_strength')
            if c=='magnet_pos':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets*3
                for i in range(0,self.number_of_magnets*3):
                    initial.append(0)
                    upper_bound.append(5)
                    lower_bound.append(-5)
                self.calib.append(1)
                print('\tmagnet_pos')
            if c=='magnet_angle':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets*3
                for i in range(0,self.number_of_magnets*3):
                    initial.append(0)
                    upper_bound.append(10)
                    lower_bound.append(-10)
                self.calib.append(2)
                print('\tmagnet_angle')
            if c=='sensor_pos':
                self.number_of_parameters = self.number_of_parameters+self.number_of_sensors*3
                for i in range(0,self.number_of_sensors*3):
                    initial.append(0)
                    upper_bound.append(5)
                    lower_bound.append(-5)
                self.calib.append(3)
                print('\tsensor_pos')
            if c=='sensor_angle':
                self.number_of_parameters = self.number_of_parameters+self.number_of_sensors*3
                for i in range(0,self.number_of_sensors*3):
                    initial.append(0)
                    upper_bound.append(10)
                    lower_bound.append(-10)
                self.calib.append(4)
                print('\tsensor_angle')

        print('number_of_magnets: %d\nnumber_of_sensors: %d\nnumber_of_parameters: %d'\
            %(self.number_of_magnets,self.number_of_sensors,self.number_of_parameters))
        res = least_squares(self.calibrationFunc, initial, bounds = (lower_bound, upper_bound), \
                            ftol=1e-8, \
                            xtol=1e-8,verbose=2, \
                            max_nfev=self.config['calibration']['max_nfev'])

        field_strength = self.config['field_strength']
        magnet_pos = self.config['magnet_pos']
        magnet_angle = self.config['magnet_angle']

        sensor_pos = self.config['sensor_pos']
        sensor_angle = self.config['sensor_angle']

        field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets = self.decodeCalibrationX(res.x)

        sensors = self.gen_sensors_all(sensor_pos,sensor_pos_offsets,sensor_angle,sensor_angle_offsets)

        print("b_field_error with calibration: %f\n"%self.calibrationFunc(res.x)[0])

        j = 0
        for target,pos,angle in zip(self.sensor_values,self.config['calibration']['magnet_pos'],self.config['calibration']['magnet_angle']):
            print("target b_field for magnet pos %f %f %f magnet angle %f %f %f"%(pos[0],pos[1],pos[2],angle[0],angle[1],angle[2]))
            i = 0
            for sens in sensors:
                print('%.4f    %.4f    %.4f'%(target[i][0],target[i][1],target[i][2]))
                i = i+1
            print("b_field with calibration:")
            magnets = self.gen_magnets_all(field_strength,[pos],magnet_pos_offsets,[angle],magnet_angle_offsets)
            for sens in sensors:
                mag = sens.getB(magnets)
                print('%.4f    %.4f    %.4f'%(mag[0],mag[1],mag[2]))
            print('----------------------------------')
            j = j+1

        print("\noptimization results:\n")
        for c in self.calib:
            if c==0:
                print('field_strength')
                print(field_strength)
                self.config['field_strength'] = field_strength
            if c==1:
                print('magnet_pos_offsets')
                print(magnet_pos_offsets)
                self.config['magnet_pos_offsets'] = magnet_pos_offsets
            if c==2:
                print('magnet_angle_offsets')
                print(magnet_angle_offsets)
                self.config['magnet_angle_offsets'] = magnet_angle_offsets
            if c==3:
                print('sensor_pos_offsets')
                print(sensor_pos_offsets)
                self.config['sensor_pos_offsets'] = sensor_pos_offsets
            if c==4:
                print('sensor_angle_offsets')
                print(sensor_angle_offsets)
                self.config['sensor_angle_offsets'] = sensor_angle_offsets

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        sensor_log_file = timestamp+'.log'
        with open(sensor_log_file, 'w') as file:
            documents = dump(sensor_log, file)
        print('sensor log written to '+sensor_log_file)
        input("Enter to write optimization to config file %s..."%(self.config_file))

        with open(self.config_file, 'w') as file:
            documents = dump(self.config, file)

    def visualizeCloud(self,mag_values,pos_values,scale):
        cloud = pcl.PointCloud_PointXYZRGB()
        number_of_samples = len(pos_values)
        points = np.zeros((number_of_samples*2, 4), dtype=np.float32)
        i = 0
        for pos,mag in zip(pos_values,mag_values):
            # dir = mag/np.linalg.norm(mag)
            p = (pos+scale*mag)/100.0

            points[i][0] = p[0]
            points[i][1] = p[1]
            points[i][2] = p[2]
            if np.linalg.norm(p)>0.22:
                points[i][3] = 255 << 16 | 255 << 8 | 255
            else:
                points[i][3] = 0 << 16 | 0 << 8 | 255

            if i%10==0:
                points[number_of_samples+i][0] = pos[0]/100.0
                points[number_of_samples+i][1] = pos[1]/100.0
                points[number_of_samples+i][2] = pos[2]/100.0
                points[number_of_samples+i][3] = 255 << 16 | 0 << 8 | 255
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
        # print('\ncomparisons %d'%comparisons)
        # print('approx time %d seconds or %f minutes'%(comparisons/1283370,comparisons/1283370/60))
        # timestamp = time.strftime("%H:%M:%S")
        # print('start time: %s'%timestamp)
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
        # print('actual time: %d'%(end - start))
        # print('\ncollisions: %d'%collisions)
        # print('average magnetic_field_difference: %f\n'%(magnetic_field_difference/comparisons))
        # sort by magnetic field differences, from low to high
        diffs, coll = zip(*sorted(zip(magnetic_field_differences, colliders)))
        return coll,diffs

    def calculateCollisionsCUDA(self,sensor_values):
        number_of_samples = len(sensor_values)
        p1 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
        p2 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
        p3 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
        p4 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
        i = 0
        for val in sensor_values:
            p1[i] = val[0]
            p2[i] = val[1]
            p3[i] = val[2]
            p4[i] = val[3]
            i = i+1
        number_of_samples = np.int32(number_of_samples)
        distance(number_of_samples, p1_gpu, p2_gpu, p3_gpu, p4_gpu, out_gpu, block=(len(p1),len(p1),1), grid=(1,1))
        out = np.reshape(out_gpu.get(),(number_of_samples,number_of_samples))
        print(out)

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

    def gen_sensors_custom(self,positions,angles):
        sensors = []
        for pos,angle in zip(positions, angles):
            s = Sensor(pos=(pos[0],pos[1],pos[2]))
            s.rotate(angle=angle[0],axis=(1,0,0))
            s.rotate(angle=angle[1],axis=(0,1,0))
            s.rotate(angle=angle[2],axis=(0,0,1))
            sensors.append(s)
        return sensors

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
    def gen_sensors_all(self,positions,pos_offsets,angles,angle_offsets):
        sensors = []
        for pos,pos_offset,angle,angle_offset in zip(positions,pos_offsets,angles,angle_offsets):
            s = Sensor(pos=(pos[0]+pos_offset[0],pos[1]+pos_offset[1],pos[2]+pos_offset[2]))
            s.rotate(angle=angle[0]+angle_offset[0],axis=(1,0,0))
            s.rotate(angle=angle[1]+angle_offset[1],axis=(0,1,0))
            s.rotate(angle=angle[2]+angle_offset[2],axis=(0,0,1))
            sensors.append(s)
        return sensors
    def gen_magnets_custom(self,positions,angles):
        magnets = []
        for field,mag_dim,pos,angle in zip(self.config['field_strength'],\
            self.config['magnet_dimension'],positions,angles):
            magnet = Box(mag=(0,0,field), dim=mag_dim,\
                pos=(pos[0],pos[1],pos[2]))
            magnet.rotate(angle=angle[0],axis=(1,0,0))
            magnet.rotate(angle=angle[1],axis=(0,1,0))
            magnet.rotate(angle=angle[2],axis=(0,0,1))
            magnets.append(magnet)
        return Collection(magnets)
    def gen_magnets_angle(self,angles):
        magnets = []
        for field,mag_dim,pos,angle in zip(self.config['field_strength'],\
            self.config['magnet_dimension'],self.config['magnet_pos'],angles):
            magnet = Box(mag=(0,0,field), dim=mag_dim,\
                pos=(pos[0],pos[1],pos[2]))
            magnet.rotate(angle=angle[0],axis=(1,0,0))
            magnet.rotate(angle=angle[1],axis=(0,1,0))
            magnet.rotate(angle=angle[2],axis=(0,0,1))
            magnets.append(magnet)
        return Collection(magnets)
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
    def gen_magnets_all(self,field_strength,positions,position_offsets,angles,angle_offsets):
        magnets = []
        i = 0
        for field,mag_dim,pos,pos_offset,angle,angle_offset in zip(field_strength,\
                self.config['magnet_dimension'],positions,position_offsets,\
                angles,angle_offsets):
            magnet = Box(mag=(0,0,field), dim=mag_dim,\
                pos=(pos[0]+pos_offset[0],\
                    pos[1]+pos_offset[1],\
                    pos[2]+pos_offset[2])\
                    )
            magnet.rotate(angle=angle[0]+angle_offset[0],axis=(1,0,0))
            magnet.rotate(angle=angle[1]+angle_offset[1],axis=(0,1,0))
            magnet.rotate(angle=angle[2]+angle_offset[2],axis=(0,0,1))
            magnets.append(magnet)
            i = i+1
        return Collection(magnets)
    def plotMagnets(self,magnets):
        for sens in self.sensors:
            print(sens.getB(magnets))
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
