import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')


def animate(i):
    x = np.linspace(0, 4, 50)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    # data = np.hstack((x[:, np.newaxis], y[:, np.newaxis], np.zeros((len(x), 1))))
    data = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    scat.set_offsets(data)
    return scat,


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((-1, 4))
    ax.set_ylim((-2, 2))
    scat = ax.scatter([], [], s=100)

    anim = FuncAnimation(fig, animate,
                         frames=200, interval=20, blit=True)
    plt.show()
    # anim.save('sine_wave.gif', writer='imagemagick')
    # anim.save('sine_wave.mp4', writer='ffmpeg')
