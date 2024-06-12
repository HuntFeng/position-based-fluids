# Position Based Fluids Miles Macklin  Matthias Muller
# https://mmacklin.com/pbf_sig_preprint.pdf
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

N = 1000
h = 1.1
L = ti.Vector([80, 50])  # the region must be wide enough to settle fluid
dim = len(L)
mass = 1
rho0 = 1
g = 9.8
dt = 1 / 20
cell_size = 2.5
screen_res = (800, 500)
screen_to_world_ratio = screen_res[0] / L[0]
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio


r_old = ti.Vector.field(dim, float, shape=(N))
r = ti.Vector.field(dim, float, shape=(N))
v = ti.Vector.field(dim, float, shape=(N))
a = ti.Vector.field(dim, float, shape=(N))
rho = ti.field(float, shape=(N))
m = ti.field(float, shape=(N))

# index of neighbor particles of each particle
num_cells = ti.Vector([int(L[d] / cell_size) for d in range(dim)])
max_num_neighbors = int(N * 0.1)
particle_neighbor = ti.field(int, shape=(N, max_num_neighbors))
# particle_at_cell[lci,lcj, :] -> indexes of particles at cell with logical coord (lci, icj)
particle_at_cell = ti.field(int, shape=(num_cells[0], num_cells[1], max_num_neighbors))
# number of neighor particles at a grid cell
grid_num_particles = ti.field(int, shape=num_cells)
lambdas = ti.field(float, shape=(N))
# use to correct position by density constraint
delta_r = ti.Vector.field(dim, float, shape=(N))


@ti.func
def w(dr) -> float:
    """general purpose kernel function"""
    poly6_factor = 315.0 / 64.0 / tm.pi
    s = dr.norm()
    result = 0.0
    if 0 <= s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def dw(dr) -> ti.Vector:
    """spiky_grad for pressure computation"""
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
        r_old[i] = ti.Vector([ti.random(float) * L[0] / 2, ti.random(float) * L[1]])
        v[i] = ti.Vector([0.0, 0.0])
        a[i] = ti.Vector([0.0, -g])  # with gravity
        rho[i] = 0
        m[i] = mass


@ti.func
def logical_coord(pos) -> ti.Vector:
    """logical coordinate of i-th particle"""
    return ti.cast(pos / cell_size, int)


@ti.func
def is_in_grid(lc) -> bool:
    """check if a logical coordinate is in the grid"""
    result = True
    for d in range(dim):
        result = result and 0 <= lc[d] and lc[d] < num_cells[d]
    return result


@ti.kernel
def find_neighbor_particles():
    """find neighbor particles based on predicted position"""
    # clear the lookup table
    for I in ti.grouped(particle_at_cell):
        particle_at_cell[I] = -1

    for I in ti.grouped(particle_neighbor):
        particle_neighbor[I] = -1

    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0

    # put each particle into their corresponding cell
    for i in r:
        lc = logical_coord(r[i])
        if grid_num_particles[lc] < max_num_neighbors:
            nb = ti.atomic_add(grid_num_particles[lc], 1)
            particle_at_cell[lc, nb] = i

    # for each particle, save particles in neighboring cells
    for i in r:
        lc = logical_coord(r[i])
        nb = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell = lc + offs
            if is_in_grid(cell):
                for k in range(max_num_neighbors):
                    particle_index = particle_at_cell[cell, k]
                    if particle_index == -1:
                        break
                    particle_neighbor[i, nb] = particle_index
                    nb += 1


@ti.kernel
def calculate_density():
    for i in r:
        rho[i] = 0.0
        for k in range(max_num_neighbors):
            j = particle_neighbor[i, k]
            if j == -1:
                break
            rho[i] += m[j] * w(r[i] - r[j])


@ti.kernel
def calculate_lambda():
    epsilon = 100.0
    for i in r:
        C_i = rho[i] / rho0 - 1
        dC_i = ti.Vector([0.0, 0.0])
        dC_norm_sqr_sum = 0.0
        for k in range(max_num_neighbors):
            j = particle_neighbor[i, k]
            if j == -1:
                break
            dC_j = dw(r[i] - r[j])
            dC_i += dC_j
            dC_norm_sqr_sum += dC_j.norm_sqr()
        dC_norm_sqr_sum += dC_i.norm_sqr()
        lambdas[i] = -C_i / (dC_norm_sqr_sum + epsilon)


@ti.func
def s_corr(i: int, j: int) -> float:
    """artificial pressure term to prevent tensile instability"""
    k = 0.001
    n = 4
    dq = ti.Vector([0.3 * h, 0])  # such that dq_norm = 0.1h
    return -k * (w(r[i] - r[j]) / w(dq)) ** n


@ti.kernel
def calculate_delta_r():
    for i in r:
        delta_r[i] = ti.Vector([0.0, 0.0])
        for k in range(max_num_neighbors):
            j = particle_neighbor[i, k]
            if j == -1:
                break
            delta_r[i] += (
                (lambdas[i] + lambdas[j] + s_corr(i, j)) * dw(r[i] - r[j]) / rho0
            )


@ti.func
def confine_to_boundary(pos) -> ti.Vector:
    epsilon = 1e-5
    bd_min = ti.Vector([0.0, 0.0]) + particle_radius_in_world
    bd_max = L - particle_radius_in_world
    for d in range(dim):
        if pos[d] <= bd_min[d]:
            pos[d] = bd_min[d] + epsilon * ti.random()

        if pos[d] >= bd_max[d]:
            pos[d] = bd_max[d] - epsilon * ti.random()
    return pos


@ti.kernel
def predict_particles():
    for i in r:
        v[i] += a[i] * dt
        r[i] = confine_to_boundary(r_old[i] + v[i] * dt)


@ti.kernel
def correct_positions():
    for i in r:
        # r[i] cannot appear on both sides of assignment statement
        # otherwise will case data racing (read and write)
        new_pos = r[i] + delta_r[i]
        r[i] = confine_to_boundary(new_pos)


@ti.kernel
def apply_viscosity():
    """apply XSPH viscosity"""
    c = 0.01
    for i in v:
        # have to use temporary varialble
        # since v[i] cannot appear on both sides of assignment statement
        temp = ti.Vector([0.0, 0.0])
        for k in range(max_num_neighbors):
            j = particle_neighbor[i, k]
            if j == -1:
                break
            temp += c * (v[j] - v[i]) * w(r[i] - r[j])
        v[i] += temp


@ti.kernel
def update_particles():
    for i in r:
        v[i] = (r[i] - r_old[i]) / dt
        r_old[i] = r[i]


def advance():
    predict_particles()
    find_neighbor_particles()
    for _ in range(5):
        calculate_density()
        calculate_lambda()
        calculate_delta_r()
        correct_positions()
    update_particles()
    apply_viscosity()


@ti.kernel
def apply_external_force(mousex: float, mousey: float):
    force_center = ti.Vector([L[0] * mousex, L[1] * mousey])
    for i in r:
        dr = r[i] - force_center
        dist = dr.norm() + 0.01
        force = 0.5 * g * dr / dist  # repel away from force center
        # modify velocity rather than acceleration for better interation
        v[i] += force * dt


def render(gui):
    bg_color = 0x112F41
    boundary_color = 0xEBACA2
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


@ti.kernel
def stats():
    avg_rho = 0.0
    for i in rho:
        avg_rho += rho[i] / N
    print("average density:", avg_rho)


if __name__ == "__main__":
    init_particles()
    gui = ti.GUI("PBD", res=screen_res)
    step = 0
    while gui.running:
        advance()
        render(gui)
        gui.get_event(ti.GUI.PRESS)  # must call this to get events
        if gui.is_pressed(ti.GUI.LMB):
            mousex, mousey = gui.get_cursor_pos()
            apply_external_force(mousex, mousey)

        # if step % 100 == 0:
        #     stats()
        step += 1
