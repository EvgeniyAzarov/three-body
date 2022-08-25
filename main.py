import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

G = 6.67408e-11  # N-m2/kg2
m_nd = 1.989e+30  # kg
r_nd = 5.326e+12  # m
v_nd = 30000  # m/s
t_nd = 79.91 * 365.25 * 24 * 3600  # s

# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

n = 4
dim = 3
bodies = np.empty(n,
                  dtype=[('pos', float, (dim,)),
                         ('vel', float, (dim,)),
                         ('mass', float),
                         ('color', float, (4,)),
                         ('size', float)])
bodies['mass'] = np.array([
    100.1,
    0.907,
    0.825,
    0.7
])
bodies['size'] = bodies['mass'] * 70

bodies['color'] = [
    colors.to_rgba('darkblue'),
    colors.to_rgba('red'),
    colors.to_rgba('green'),
    colors.to_rgba('purple')
]

bodies['pos'] = np.array([
    [-0.5, 1, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0],
    [0.5, 0.5, 0.5]
])
r_com = bodies['mass'].dot(bodies['pos']) / bodies['mass'].sum()

bodies['vel'] = np.array([
    [0.01, 0.01, 0],
    [-0.05, 0, -0.1],
    [0, 0, 0],
    [0.01, -0.01, 0.01]
])
v_com = bodies['mass'].dot(bodies['vel']) / bodies['mass'].sum()


def motion_ode(t, w, m, n, dim):
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


def update(frame, ax, graphs, orbits, scat):
    j = 2 * frame + 1
    tail = 200
    start = max(0, j - tail)
    bodies['pos'] = orbits[j - 1]

    for i in range(n):
        graphs[i].set_data(*orbits[start:j, i, 0:2].T)
        graphs[i].set_3d_properties(orbits[start:j, i, 2])

    # print(orbits[start:j, :, :])
    ax.set_xlim(orbits[0:j, :, 0].min(), orbits[0:j, :, 0].max())
    ax.set_ylim(orbits[0:j, :, 1].min(), orbits[0:j, :, 1].max())
    ax.set_zlim(orbits[0:j, :, 2].min(), orbits[0:j, :, 2].max())

    # scat._offsets3d = bodies['pos'].T
    scat[0].remove()
    scat[0] = ax.scatter(*bodies['pos'].T, s=bodies['size'], c=bodies['color'], depthshade=False)


def main():
    w_init = np.concatenate((bodies['pos'].flatten(), bodies['vel'].flatten()))
    time_span = np.linspace(0, 12, 1000)
    # w = euler_ode_solver(motion_ode, w_init, time_span, m=bodies['mass'], n=n, dim=dim)
    # sol = sci.integrate.solve_ivp(motion_ode, (0, 10), w_init, t_eval=time_span,
    #                               args=(bodies['mass'], n, dim))
    # w = sol.y
    w = sci.integrate.odeint(motion_ode, w_init, time_span, args=(bodies['mass'], n, dim), tfirst=True)
    w = np.array(w)
    print(w)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    scat = ax.scatter(*bodies['pos'].T, s=bodies['size'], c=bodies['color'], depthshade=False)

    orbits = w[:, :dim * n].reshape((-1, n, dim))

    # print(orbits.shape)
    # print(orbits_com.shape)

    # orbits -= orbits_com[:, :, None]

    graphs = []
    for i in range(n):
        graph, = ax.plot(*orbits[:1, i].T, color=bodies['color'][i])
        graphs.append(graph)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    # ax.set_title("visualization of orbits of stars in a two-body system\n", fontsize=14)
    # ax.legend(loc="upper left", fontsize=14)

    anim = animation.FuncAnimation(fig, update, interval=10, frames=400, repeat=False,
                                   fargs=(ax, graphs, orbits, [scat]))
    # anim.save('out/three_bodies.mp4')
    anim.save('out/three_bodies.gif')

    plt.show()


if __name__ == '__main__':
    main()
