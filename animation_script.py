import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import csv
import numpy as np
import pandas as pd
import matplotlib.cm as cm

trial_name = 'right_trim'

# Load in tracked joint data from 3D pose and pass to array (XYZ)
data_x = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_x_no_headers.csv')
pose_x = np.array(data_x, dtype='float')
data_y = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_y_no_headers.csv')
pose_y = np.array(data_y, dtype='float')
data_z = pd.read_csv('./Out/Data/' + trial_name + '_3Dtracked_z_no_headers.csv')
pose_z = np.array(data_z, dtype='float')

data_frames, data_joints = pose_x.shape
track_marker_idx = 29

# Generate empty arrays for ploting joint paths
X = []
Y = []
Z = []

# Select colour map
colours = cm.prism(np.linspace(0, 1 , data_joints))

# Plotting update function to iterate through frames
def update(i):
    ax.cla()

    x = pose_x[i, :]
    y = pose_z[i, :]
    z = -pose_y[i, :]

    X.append(x[track_marker_idx])
    Y.append(y[track_marker_idx])
    Z.append(z[track_marker_idx])

    ax.scatter(x, y, z, c = colours, s = 10, marker = 'o')

    for i_joint in range(data_joints):
        ax.plot(X, Y, Z, c = colours[i_joint])    

    ax.set_xlim(-1500, 1500)
    ax.set_ylim(2000, 5000)
    ax.set_zlim(-1500, 1500)


fig = plt.figure(dpi=100)
fig.set_figheight(9.6)
fig.set_figwidth(12.8)
ax = fig.add_subplot(projection='3d')

ani = animation.FuncAnimation(fig = fig, func = update, frames = 2000, interval = 5, repeat = False)

writer = animation.PillowWriter(fps = 30,
                                    metadata = 'None',  #dict(artist = 'Me')
                                    bitrate = 'None')   #1800
ani.save(trial_name + '.gif', writer = writer )

plt.show()