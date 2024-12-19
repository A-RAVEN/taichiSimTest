import taichi as ti
ti.init(arch=ti.vulkan)



int3 = ti.types.vector(3, dtype=ti.i32)
float3 = ti.types.vector(3, float)

dimension = int3([64, 64, 64])
gravity = float3([0, -9.8, 0])

particle_dim = [32, 32, 32]
particle_count = particle_dim[0] * particle_dim[1] * particle_dim[2]

@ti.dataclass
class Particle:
    pos: float3
    vel: float3
    @ti.func
    def initialize(self, inpos: float3, invel: float3):
        self.pos=inpos
        self.vel=invel
    @ti.func
    def advect(self, delta_time: float):
        self.pos += self.vel * delta_time

particles = Particle.field(shape=particle_count)

@ti.dataclass
class Grid:
    vel: float3
    grid_type: int

grids = Grid.field(shape=dimension)

@ti.data_oriented
class GridField:
    grids
    grid_dimension: int3
    grid_width: float
    
    @ti.kernel
    def initialize_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].vel = float3([0, 0, 0])

    def __init__(self, dim: int3, gridwidth: float = 1.0):
        self.grid_width = gridwidth
        self.grid_dimension = dim
        self.grids = Grid.field(shape=self.grid_dimension)
        self.initialize_grids()

    @ti.kernel
    def update_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].vel = float3([1, 0, 0])


field=GridField(dimension)

@ti.func
def convertID3(id3, dim):
    return id3[2] * dim[0] * dim[1] + id3[1] * dim[0] + id3[0]

@ti.kernel
def initialize_grid():
    for xx, yy, zz in ti.ndrange(dimension[0], dimension[1], dimension[2]):
        grids[xx, yy, zz].vel = ti.Vector([0, 0, 0])


@ti.kernel
def initialize_particles():
    for xx, yy, zz in ti.ndrange(particle_dim[0], particle_dim[1], particle_dim[2]):
        particles[convertID3([xx, yy, zz], particle_dim)].initialize(ti.Vector([xx,yy,zz], dt=float), ti.Vector([0.0, -9.0, 0.0], dt=float))

@ti.kernel
def advec_particles(dt: float):
    for particle_id in ti.ndrange(particle_count):
        particles[particle_id].advect(dt)


initialize_grid()
initialize_particles()

positions = ti.Vector.field(3, dtype=float, shape=particle_count)
colors = ti.Vector.field(3, dtype=float, shape=particle_count)


@ti.kernel
def prepare_grid_positions(scale: float, bias: ti.template()):
    for particle_id in ti.ndrange(particle_count):
        positions[particle_id] = particles[particle_id].pos * scale + bias
        colors[particle_id] = ti.Vector([0, 0, 1], dt=float)

localbias = ti.Vector([-0.5, -0.5, -0.5], dt=ti.f32)

window = ti.ui.Window("Test 3D Grid Simulation", (1920, 1080), vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 1.0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

deltatime = 1.0 / 60.0
while window.running:
    field.update_grids()
    advec_particles(deltatime)
    prepare_grid_positions(0.01, localbias)
    camera.position(3, 3, 3)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light([0.5, 0.5, 0.5])
    scene.particles(positions, radius=0.005, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()