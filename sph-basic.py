# Smoothed Particle Hydrodynamics: Theory, Implementation, and Application to Toy Stars Philip Mocz
# https://pmocz.github.io/manuscripts/pmocz_sph.pdf
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

N = 1000
h = 0.08
L = ti.Vector([1, 1])
dim = len(L)
V = np.prod(L)
mass = 1
rho0 = N * mass / V * 2
g = 9.8
dt = 0.002
screen_res = (500, 500)
screen_to_world_ratio = screen_res[0] / L[0]


r = ti.Vector.field(dim, float, shape=(N))
v = ti.Vector.field(dim, float, shape=(N))
a = ti.Vector.field(dim, float, shape=(N))
rho = ti.field(float, shape=(N))
p = ti.field(float, shape=(N))
m = ti.field(float, shape=(N))


@ti.func
def w(dr) -> float:
    # general purpose kernel function
    poly6_factor = 315.0 / 64.0 / tm.pi
    s = dr.norm()
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def dw(dr) -> ti.Vector:
    # spiky_grad for pressure computation
    spiky_grad_factor = -45.0 / tm.pi
    r_len = dr.norm()
    result = ti.Vector([0.0, 0.0])
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = dr * g_factor / r_len
    return result


@ti.func
def ddw(dr) -> float:
    # return -2 / h**2 * (dim * w(dr) + tm.dot(dr, dw(dr)))
    # for viscosity
    factor = 45 / (tm.pi * h**6)
    norm = dr.norm()
    result = 0.0
    if norm < h:
        result = factor * (h - norm)
    return result


@ti.kernel
def init_particles():
    for i in r:
        r[i] = ti.Vector([ti.random(float) * L[0], ti.random(float) * L[1] / 2])
        v[i] = ti.Vector([0.0, 0.0])
        a[i] = ti.Vector([0.0] * dim)
        rho[i] = 0
        p[i] = 0
        m[i] = mass


@ti.kernel
def calculate_density():
    for i in rho:
        rho[i] = 0
        for j in range(N):
            rho[i] += m[j] * w(r[i] - r[j])


@ti.kernel
def calculate_pressure():
    k = 0.5
    for i in p:
        p[i] = k * (rho[i] - rho0)


@ti.func
def grad_p_over_rho(i: int) -> ti.Vector:
    grad_p_over_rho = ti.Vector([0.0, 0.0])
    for j in range(N):
        grad_p_over_rho += (
            m[j] * (p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2) * dw(r[i] - r[j])
        )
    return grad_p_over_rho


@ti.func
def viscosity_over_rho(i: int) -> ti.Vector:
    mu = 1
    vis = ti.Vector([0.0, 0.0])
    for j in range(N):
        vis += mu * m[j] * (v[j] - v[i]) / rho[j] * ddw(r[i] - r[j])
    return vis / rho[i]


@ti.func
def surface_tension_over_rho(i: int) -> ti.Vector:
    Cs = 0.0
    for j in range(N):
        Cs += m[j] / rho[j] * w(r[i] - r[j])
    dCs = ti.Vector([0.0, 0.0])
    for j in range(N):
        dCs += m[j] * Cs / rho[j] * dw(r[i] - r[j])

    tension = ti.Vector([0.0, 0.0])
    norm = dCs.norm()
    if norm > 1e-3:
        sigma = 0.1
        ddCs = 0.0
        for j in range(N):
            ddCs += m[j] * Cs / rho[j] * ddw(r[i] - r[j])
        tension = -sigma * ddCs * dCs / norm
    return tension / rho[i]


@ti.kernel
def calculate_acceleration():
    for i in a:
        grav = ti.Vector([0, -g])
        grad_p = grad_p_over_rho(i)
        viscosity = viscosity_over_rho(i)
        tension = surface_tension_over_rho(i)
        a[i] = -grad_p + viscosity + tension + grav


@ti.kernel
def push_particles():
    for i in r:
        v[i] += a[i] * dt
        r[i] += v[i] * dt


@ti.kernel
def apply_boundary():
    for i in r:
        for d in range(dim):
            if r[i][d] <= 0:
                r[i][d] *= -1
                v[i][d] *= -1

            bd = tm.max(1, L[d])
            if r[i][d] >= bd:
                r[i][d] = 2 * bd - r[i][d]
                v[i][d] *= -1


def advance():
    calculate_density()
    calculate_pressure()
    calculate_acceleration()
    push_particles()
    apply_boundary()


def render(gui):
    bg_color = 0x112F41
    boundary_color = 0xEBACA2
    particle_radius = 4
    particle_color = 0x068587
    gui.clear(bg_color)

    pos_np = r.to_numpy()
    # print(pos_np)
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect(
        (0, 0),
        (1, 1),
        radius=1.5,
        color=boundary_color,
    )
    gui.show()


if __name__ == "__main__":
    init_particles()
    gui = ti.GUI("SPH", res=screen_res)
    while gui.running:
        advance()
        render(gui)
