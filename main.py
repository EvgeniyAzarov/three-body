import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2

# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri

# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

n = 2
dim = 3
bodies = np.empty(n,
                  dtype=[('pos', float, (dim,)),
                         ('vel', float, (dim,)),
                         ('mass', float),
                         ('color', float, (4,)),
                         ('size', float)])
bodies['mass'] = np.array([
    1.1,
    0.907,
    # 0.2
])
bodies['size'] = bodies['mass'] * 10
bodies['color'] = [
    colors.to_rgba('darkblue'),
    colors.to_rgba('red'),
    # colors.to_rgba('green')
]

bodies['pos'] = np.array([
    [-0.05, 0, 0],
    [0.05, 0, 0],
    # [0, 0.5, 0]
])
r_com = bodies['mass'].dot(bodies['pos']) / bodies['mass'].sum()

bodies['vel'] = np.array([
    [0.01, 0.01, 0],
    [-0.05, 0, -0.1],
    # [0, 0, 0]
])
v_com = bodies['mass'].dot(bodies['vel']) / bodies['mass'].sum()


def motion_ode(t, w, m, n=3, dim=3):
    r = w[: n * dim]
    v = w[n * dim:]

    # print(r)
    # print(v)
    # print()

    rr = r.reshape((n, dim))
    rij = rr[None, :, :] - rr[:, None, :]
    dists = np.linalg.norm(rij, axis=2) + np.diag(np.ones(n))

    rij_adj = rij / dists[:, :, None] ** 3

    dvdt = (K1 * rij_adj * m[:, None]).sum(axis=1).flatten()
    drdt = K2 * v

    derivatives = np.concatenate((drdt, dvdt))

    return derivatives


def euler_ode_solver(derivs, w_init, t, **kwargs):
    w = [w_init, ]
    for i in range(1, len(t)):
        # print(w[-1])
        w.append(w[-1] + (t[i] - t[i - 1]) * derivs(w[-1], **kwargs))

    return np.array(w)


def update(frame, graphs, orbits):
    for i in range(n):
        graphs[i].set_data(*orbits[:frame, i, 0:2].T)
        graphs[i].set_3d_properties(orbits[:frame, i, 2])


def main():
    w_init = np.concatenate((bodies['pos'].flatten(), bodies['vel'].flatten()))
    time_span = np.linspace(0, 10, 5000)
    w = euler_ode_solver(motion_ode, w_init, time_span, m=bodies['mass'], n=n, dim=dim)
    # sol = sci.integrate.solve_ivp(motion_ode, (0, 10), w_init, t_eval=time_span,
    #                               args=(bodies['mass'], n, dim))
    # w = sol.y
    print(w)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    scat = ax.scatter(*bodies['pos'].T, s=bodies['size'], c=bodies['color'], depthshade=False)

    orbits = w[:, :dim * n].reshape((-1, n, dim))

    graphs = []
    for i in range(n):
        graph, = ax.plot(*orbits[:1, i].T, color=bodies['color'][i])
        graphs.append(graph)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    # ax.set_title("visualization of orbits of stars in a two-body system\n", fontsize=14)
    # ax.legend(loc="upper left", fontsize=14)

    anim = animation.FuncAnimation(fig, update, interval=50, repeat=False, fargs=(graphs, orbits))
    anim.save('two_bodies.mp4')
    anim.save('two_bodies.gif')

    # plt.show()


if __name__ == '__main__':
    main()
