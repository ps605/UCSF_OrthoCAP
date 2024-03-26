import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.animation as animation
import csv
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import os

# SETUP
plt.ioff()
flag_seperateXYZ    = False
flag_makeGIF        = True
flag_midShldrPevlis = False

# Where to read data from
data_path = '../Study_XSens_Ames/In/Box_data/'

# List files in directory, loop through them and check for .csv
csv_files = os.listdir(data_path)

for csv_file in csv_files:
    if csv_file.endswith('pos.csv'):

        # Get base trial name
        trial_name = data_path + csv_file[:-8] #'IMU_Segment_pos_xyz'

        # Load in tracked joint data from 3D pose and pass to array (XYZ)
        if flag_seperateXYZ == True:
            data_x = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_x_no_headers.csv')
            pose_x = np.array(data_x, dtype='float')
            data_y = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_y_no_headers.csv')
            pose_y = np.array(data_y, dtype='float')
            data_z = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_z_no_headers.csv')
            pose_z = np.array(data_z, dtype='float')
        else:

            # Get position
            data_xyz = pd.read_csv(trial_name + '_pos.csv')
            data_xyz = data_xyz.drop(columns='Frame')
            # TO np.array and into mm
            pose_xyz = np.array(data_xyz, dtype='float')*1000

            # split data into X, Y, Z
            pose_x = pose_xyz[:,0::3]
            x_min = np.min(pose_x)
            x_max = np.max(pose_x)
            
            pose_y = pose_xyz[:,1::3]
            y_min = np.min(pose_y)
            y_max = np.max(pose_y)

            pose_z = pose_xyz[:,2::3]
            z_min = np.min(pose_z)
            z_max = np.max(pose_z)

            # Get quaternion .csv     
            # Load in tracked joint data from 3D orientation and pass to array (quaternions)    
            data_q0123 = pd.read_csv(trial_name + '_qua.csv')  
            data_q0123 = data_q0123.drop(columns='Frame')      

            # Get indeces of Pelvis and L/R Shoulders
            idx_pelvis_q = int(data_q0123.columns.get_loc('Pelvis q0')/4)
            idx_shld_r_q = int(data_q0123.columns.get_loc('Right Shoulder q0')/4)
            idx_shld_l_q = int(data_q0123.columns.get_loc('Left Shoulder q0')/4)

            # To np.array quaternion
            ori_quat = np.array(data_q0123, dtype='float')

            # split data into q0, q1, q2, q3
            # MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion
            ori_q0 = ori_quat[:,0::4]                       
            ori_q1 = ori_quat[:,1::4] 
            ori_q2 = ori_quat[:,2::4]
            ori_q3 = ori_quat[:,3::4]

            # Check both position and quat have the same frames        
            n_frames_pos, n_cols_pos = np.shape(data_xyz)
            n_frames_quat, n_cols_quat = np.shape(data_q0123)

            if n_frames_pos == n_frames_quat:
                n_frames = n_frames_pos
            else:
                n_frames = np.min([n_frames_pos, n_frames_quat])

        # Calculate delta(mid shoulder, pelvis)
        if flag_midShldrPevlis == True:
            # ! should find better way of indexing data with header info from .mvn
            pelvis = pose_xyz[:,0:3]
            shoulder_r = pose_xyz[:,21:24]
            shoulder_l = pose_xyz[:,33:36]
            # Calculate vector between pelvis and mid shoulders (in global frame)
            d_mShd2Pel = np.mean(np.array([shoulder_r, shoulder_l]), axis=0) - pelvis

        n_frames, n_cols = np.shape(pose_xyz)
        frames_v = range(n_frames)
      

        # plt.figure()
        # if flag_midShldrPevlis == True:
        #     plt.scatter(d_mShd2Pel[:,0], d_mShd2Pel[:,1], c=frames_v[:]], cmap='jet')
        #     plt.colorbar()
        # plt.xlim(-150, 150)
        # plt.ylim(-150, 150)
        # plt.xlabel('Anterior displacement (mm)')
        # plt.ylabel('Lateral displacement (mm)')        
        # plt.savefig(data_path + 'Figures/' + csv_file[0:-4] + '.png')
        # plt.close()

        # Generate .gif of motion
        if flag_makeGIF:
            data_frames, data_joints = pose_x.shape
            track_marker_idx = []

            # Generate empty arrays for ploting joint paths
            X = []
            Y = []
            Z = []

            # Select colour map
            colours = cm.prism(np.linspace(0, 1 , data_joints))

            # Plotting update function to iterate through frames
            def update(i):
                ax.cla()

                # Update position of segments
                x = pose_x[i, :]
                y = pose_y[i, :]
                z = pose_z[i, :]

                ax.scatter(x, y, z, c = 'red', s = 14, marker = 'o')

                # Orientation of pelvis in global from IMU quaternions
                rm_pelvis = R.from_quat([ori_q1[i, idx_pelvis_q], ori_q2[i, idx_pelvis_q], ori_q3[i, idx_pelvis_q], ori_q0[i, idx_pelvis_q]])
                # Get as rotation matrix to calculate vectors
                rm_pelvis_mat = rm_pelvis.as_matrix()
                # Get pelvis vectors for coordinate system plotting
                pelvis_pos = [pose_x[i,idx_pelvis_q], pose_y[i,idx_pelvis_q], pose_z[i,idx_pelvis_q]]
                pelvis_coords = rm_pelvis_mat*100 + pelvis_pos

                # Plot pelvis coordinate system
                ax.plot([pelvis_pos[0],pelvis_coords[0,0]], [pelvis_pos[1],pelvis_coords[0,1]], [pelvis_pos[2],pelvis_coords[0,2]],color = 'green')
                ax.plot([pelvis_pos[0],pelvis_coords[1,0]], [pelvis_pos[1],pelvis_coords[1,1]], [pelvis_pos[2],pelvis_coords[1,2]],color = 'red')
                ax.plot([pelvis_pos[0],pelvis_coords[2,0]], [pelvis_pos[1],pelvis_coords[2,1]], [pelvis_pos[2],pelvis_coords[2,2]],color = 'blue')
                
                
                for i_joint in track_marker_idx:
                    X.append(x[i_joint])
                    Y.append(y[i_joint])
                    Z.append(0)
                    ax.plot(X, Y, Z, c = colours[i_joint])      

                # ax.set_xlim(0, 5500)
                # ax.set_ylim(-1000, 1000)
                # ax.set_zlim(-100, 2000)
                
                ax.set_title(['Frame Number: ' + str(i)])
                ax.set_xlim(x_min + 0.1*x_min, x_max + 0.1*x_max)
                ax.set_ylim(y_min + 0.1*y_min, y_max + 0.1*y_max)
                ax.set_zlim(z_min + 0.1*z_min, z_max + 0.1*z_max)
                ax.set_aspect('equal')   
                
            fig = plt.figure(dpi=100)
            fig.set_figheight(9.6)
            fig.set_figwidth(12.8)
            ax = fig.add_subplot(projection='3d')
            
            ani = animation.FuncAnimation(fig = fig, func = update, frames = n_frames, interval = 1, repeat = False)

            writer = animation.PillowWriter(fps = 30,
                                                metadata = 'None',  #dict(artist = 'Me')
                                                bitrate = 1000)   #1800
            ani.save(data_path + 'Figures/' + csv_file[0:-4] + '.gif', writer = writer )

            print('Animation complete for:' + data_path + 'Figures/' + csv_file[0:-4] + '.gif')