from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale*0.5
        line_rot = r.apply(line)
        line_plot = (line_rot + loc)
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    ax.text(*offset, name, color="k", va="center", ha="center",
            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})
# ----- NOTE ----
# MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion
# World
r0  = R.identity()
# Pelvis
# r1 = R.from_euler("XYZ", [90, 0, 0], degrees=True)
r1 = R.from_quat([-0.050500974,	0.196824644, 0.032987938, 0.978581375])
# Head
# r2 = R.from_euler("XYZ", [0, 90, 0], degrees=True) 
r2 = R.from_quat([-0.092814583,	0.393997628, -0.010802756, 0.914349289])
# r12 = r1*r2
# Inverse World to Pelvis
r1i = r1.inv()

ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
plot_rotated_axes(ax, r0, name="W", offset=(0, 0, 0))
plot_rotated_axes(ax, r1, name="P", offset=(0.273751613,	0.433289692,	1.192318577))
plot_rotated_axes(ax, r2, name="H", offset=(0.273751613,	0.433289692,	1.592318577))

p_l =[0.065692008,	0.356454372,	1.072604097]


ax.set(xlim=(-0.1, 2.5), ylim=(-1.25, 1.25), zlim=(-0.1, 2.5))
#ax.set(xticks=[-1, 0, 2.5], yticks=[-1, 0, 2.5], zticks=[-0.1, 2.5])
ax.set_aspect("equal", adjustable="box")
ax.figure.set_size_inches(6, 5)
plt.tight_layout()

plt.show()
