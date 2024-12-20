import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cuda)

int3 = ti.types.vector(3, dtype=ti.i32)
float3 = ti.types.vector(3, float)

dimension = int3([64, 64, 64])
gravity = float3([0, -9.8, 0])

@ti.func
def convertID3(id3, dim):
    return id3[2] * dim[0] * dim[1] + id3[1] * dim[0] + id3[0]

@ti.data_oriented
class BoxMesh:
    vertices=None
    indices=None
    render_vertices=None

    @ti.kernel
    def update_transform(self, scale:float3, offset:float3):
        for vert_id in ti.ndrange(8):
            self.render_vertices[vert_id] = self.vertices[vert_id] * scale + offset

    def __init__(self):
        self.vertices = float3.field(shape=8)
        self.render_vertices = float3.field(shape=8)
        self.indices=ti.field(int, 24)
        self.vertices[0]=float3([0, 0, 0])
        self.vertices[1]=float3([1, 0, 0])
        self.vertices[2]=float3([1, 0, 1])
        self.vertices[3]=float3([0, 0, 1])
        self.vertices[4]=float3([0, 1, 0])
        self.vertices[5]=float3([1, 1, 0])
        self.vertices[6]=float3([1, 1, 1])
        self.vertices[7]=float3([0, 1, 1])
        self.update_transform(float3([1, 1, 1]), float3([0, 0, 0]))
        for i in range(4):
            offset=6 * i
            lowerId = i
            higherId = i + 4
            lowerNext = (lowerId + 1) % 4
            higherNext = lowerNext + 4
            self.indices[offset]=lowerId
            self.indices[offset + 1]=lowerNext
            self.indices[offset + 2]=higherId
            self.indices[offset + 3]=higherNext
            self.indices[offset + 4]=lowerId
            self.indices[offset + 5]=higherId

    def render(self, scene:ti.ui.Scene):
        scene.lines(self.render_vertices, 5, self.indices)

boxMesh = BoxMesh()

@ti.data_oriented
class Frame:
    frame_ID=None
    def __init__(self):
        self.frame_ID=ti.field(ti.i32, shape=1)

    def next_frame(self):
        self.frame_ID[0] += 1

    @ti.func
    def get_frame(self):
        return self.frame_ID[0]

frameContext = Frame()

@ti.func
def select(cond:bool, a:ti.template(), b:ti.template()):
    res = b
    if cond:
        res = a
    return res


@ti.dataclass
class Particle:
    pos: float3
    vel: float3
    @ti.func
    def initialize(self, inpos: float3, invel: float3=float3([0, 0, 0])):
        self.pos=inpos
        self.vel=invel
    @ti.func
    def advect(self, delta_time: float, gridfiled:ti.template()):
        next_Pos = self.pos + self.vel * delta_time
        if not gridField.valid_world_position(next_Pos):
            self.vel = -self.vel
            next_Pos = self.pos + self.vel * delta_time
        self.pos = next_Pos

@ti.data_oriented
class ParticleGroup:
    particles = None
    particle_position_list = None
    particle_color_list = None
    particle_count: int

    @ti.kernel
    def initialize_particles(self, initcenter:float3, inithalfsize:float3):
        for particle_id in ti.ndrange(self.particle_count):
            rand_pos = initcenter + (float3(ti.random(float), ti.random(float), ti.random(float)) * 2.0 - 1.0) * inithalfsize
            self.particles[particle_id].initialize(rand_pos, float3([0, -(ti.random(float)*0.5 + 3.5), 0]))

    def __init__(self, initcount:int=4096, initcenter:float3=float3([0, 0, 0]), inithalfsize:float3=float3([0.1, 0.1, 0.1])):
        self.particle_count = initcount
        if self.particle_count>0:
            self.particles=Particle.field(shape=self.particle_count)
            self.initialize_particles(initcenter, inithalfsize)
            self.particle_position_list=float3.field(shape=self.particle_count)
            self.particle_color_list=float3.field(shape=self.particle_count)

    @ti.kernel
    def advec_particles(self, dt: float, gridfiled:ti.template()):
        for particle_id in ti.ndrange(self.particle_count):
            self.particles[particle_id].advect(dt, gridfiled)
    
    @ti.kernel
    def generate_position_and_color(self):
        for particle_id in ti.ndrange(self.particle_count):
            self.particle_position_list[particle_id]=self.particles[particle_id].pos
            self.particle_color_list[particle_id]=float3(0, 1, 0)
    
    def tick(self, dt:float, gridfiled):
        self.advec_particles(dt, gridfiled)
        self.generate_position_and_color()

    def render(self, scene:ti.ui.Scene):
        if(self.particle_position_list != None) and (self.particle_color_list != None):
            scene.particles(self.particle_position_list, radius=0.01, per_vertex_color=self.particle_color_list)

        


@ti.dataclass
class Grid:
    last_vel:float3
    last_weight: float
    vel: float3
    pressure: float3
    weight: float
    debug_counter: int
    @ti.func
    def initialize(self):
        self.vel = float3(0, 0, 0)
        self.last_vel = float3(0, 0, 0)
        self.pressure = float3(0, 0, 0)
        self.last_weight = 0
        self.weight = 0
        self.debug_counter = 0

    @ti.func
    def cache(self):
        self.last_vel = self.vel
        self.last_weight = self.weight
        self.vel = float3(0.0, 0.0, 0.0)
        self.weight = 0.0

    @ti.func
    def normalize_velocity(self):
        if self.weight > 0:
            self.vel /= self.weight
            # if frameContext.get_frame() == 1:
            #     print("weight:", self.weight, ";val:", self.vel)
        else:
            self.weight = 0.0
            self.vel = float3(0.0, 0.0, 0.0)
        # if(self.last_weight > 0) and (self.weight > 0):
        #     result = self.vel - self.last_vel
        #     if(frameContext.get_frame() == 1 and result[1] > 0):
        #         print("last_weight:", self.last_weight, ";last_val:", self.last_vel, ";weight:", self.weight, ";val:", self.vel, ";resultaftersolve:", result)


    @ti.func
    def delta_velocy(self):
        result=float3(0, 0, 0)
        if(self.last_weight > 0) and (self.weight > 0):
            result = self.vel - self.last_vel
        #print("result:", result)
        return result

    @ti.func
    def increment_particle(self, in_weight: float, in_vel: float3):
        ti.atomic_add(self.weight, in_weight)
        ti.atomic_add(self.vel, in_weight * in_vel)


    @ti.func
    def increment_grid(self):
        inc_res=ti.atomic_add(self.debug_counter, 1)



@ti.data_oriented
class GridField:
    grids=None
    grid_dimension: int3
    grid_width: float
    field_width: float3
    field_offset: float3
    
    occupiedFieldCount=None
    occupiedGridPositions=None
    
    @ti.kernel
    def initialize_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].initialize()

    def __init__(self, dim: int3, incenter: float3, gridwidth: float = 1.0):
        self.grid_width = gridwidth
        self.grid_dimension = dim
        self.field_width = self.grid_width * self.grid_dimension
        self.field_offset = incenter - 0.5 * self.field_width
        self.grids = Grid.field(shape=self.grid_dimension)
        self.occupiedGridPositions = float3.field(shape=self.grid_dimension[0] * self.grid_dimension[1] * self.grid_dimension[2])
        self.occupiedFieldCount = ti.field(ti.i32, shape=1)
        self.initialize_grids()

    @ti.kernel
    def cache_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].cache()

    @ti.kernel
    def solve_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].normalize_velocity()

    def draw_field_range(self, scene: ti.ui.Scene):
        boxMesh.update_transform(self.field_width, self.field_offset)
        boxMesh.render(scene)

    @ti.func
    def worldPosToGrid(self, worldPos:float3):
        return (worldPos-self.field_offset) / self.grid_width
    @ti.func
    def gridPosToWorld(self, gridPos:float3):
        return gridPos * self.grid_width + self.field_offset
    @ti.func
    def worldVecToGrid(self, worldVec:float3):
        return worldVec / self.grid_width
    @ti.func
    def gridVecToWorld(self, gridVec:float3):
        return gridVec * self.grid_width
    
    @ti.func
    def valid_local_position(self, localpos:float3):
        return (localpos[0] > 0.5 and localpos[0] < ti.cast(self.grid_dimension[0], float) - 0.5)\
            and (localpos[1] > 0.5 and localpos[1] < ti.cast(self.grid_dimension[1], float) - 0.5)\
            and (localpos[2] > 0.5 and localpos[2] < ti.cast(self.grid_dimension[2], float) - 0.5)
    
    @ti.func
    def valid_world_position(self, world_pos:float3):
        return self.valid_local_position(self.worldPosToGrid(world_pos))

    @ti.func
    def get_grid(self, grid_id: int3):
        return self.grids[grid_id[0], grid_id[1], grid_id[2]]
    
    @ti.func
    def set_grid(self, grid_id: int3, grid:Grid):
        self.grids[grid_id[0], grid_id[1], grid_id[2]] = grid

    @ti.func
    def particle_to_grid_weight(self, pt0:float3, pt1:float3):
        pt0 = tm.clamp(pt0, 0.5, ti.cast(self.grid_dimension, float) - 0.5)
        pt1 = tm.clamp(pt1, 0.5, ti.cast(self.grid_dimension, float) - 0.5)
        dist = 1.0 - abs(pt1-pt0)
        return tm.clamp((dist[0] * dist[1] * dist[2]), 0.0, 1.0)


    @ti.kernel
    def insertParticles(self, particleGroup: ti.template()):
        for particle_id in ti.ndrange(particleGroup.particle_count):
            itr_patricle = particleGroup.particles[particle_id]
            particle_localpos = self.worldPosToGrid(itr_patricle.pos)
            if self.valid_local_position(particle_localpos):
                particle_localvel = self.worldVecToGrid(itr_patricle.vel)
                delta_pos = tm.fract(particle_localpos)
                indexBias= float3([select(delta_pos[0] < 0.5, -1, 0), select(delta_pos[1] < 0.5, -1, 0), select(delta_pos[2] < 0.5, -1, 0)]) 
                base_grid_f = tm.max(tm.floor(particle_localpos)+indexBias, 0)
                base_grid_i = ti.cast(base_grid_f, ti.i32)
                for xx in range(2):
                    for yy in range(2):
                        for zz in range(2):
                            local_offset = int3(xx, yy, zz)
                            grid_id = base_grid_i + local_offset
                            grid_center = ti.cast(grid_id, float) + 0.5
                            to_grid_weight = self.particle_to_grid_weight(particle_localpos, grid_center)
                            self.get_grid(grid_id).increment_particle(to_grid_weight, particle_localvel)

    @ti.kernel
    def portBackToParticles(self, particleGroup: ti.template()):
        for particle_id in ti.ndrange(particleGroup.particle_count):
            #itr_patricle = particleGroup.particles[particle_id]
            particle_localpos = self.worldPosToGrid(particleGroup.particles[particle_id].pos)
            if self.valid_local_position(particle_localpos):
                delta_pos = tm.fract(particle_localpos)
                indexBias= float3([select(delta_pos[0] < 0.5, -1, 0), select(delta_pos[1] < 0.5, -1, 0), select(delta_pos[2] < 0.5, -1, 0)]) 
                base_grid_f = tm.max(tm.floor(particle_localpos)+indexBias, 0)
                #print("particle:", particle_localpos, ";base_id:", base_grid_f)
                base_grid_i = ti.cast(base_grid_f, ti.i32)
                combined_del_vel = float3(0, 0, 0)
                combined_vel = float3(0, 0, 0)
                combined_weight = 0.0
                for xx in range(2):
                    for yy in range(2):
                        for zz in range(2):
                            local_offset = int3(xx, yy, zz)
                            grid_id = base_grid_i + local_offset
                            grid_center = ti.cast(grid_id, float) + 0.5
                            grid_weight = self.get_grid(grid_id).weight
                            to_grid_weight = self.particle_to_grid_weight(particle_localpos, grid_center)
                            combined_del_vel += self.get_grid(grid_id).delta_velocy() * to_grid_weight * grid_weight
                            combined_vel += self.get_grid(grid_id).vel * to_grid_weight
                            combined_weight += grid_weight
                if combined_weight > 0:
                    combined_del_vel /= combined_weight
                    #combined_vel /= combined_weight
                #print("[",frameContext.get_frame(),"]","del_vel:", combined_del_vel)
                
                particle_vel = self.worldVecToGrid(particleGroup.particles[particle_id].vel)
                newVelFlip = particle_vel + combined_del_vel
                flip_str = 0.8
                particle_vel = flip_str * newVelFlip + (1.0 - flip_str) * combined_vel

                vel_length = tm.length(particle_vel)
                if vel_length > 500.0:
                    particle_vel *= 500.0 / vel_length
                #print("combined_vel,", combined_vel)
                particleGroup.particles[particle_id].vel = self.gridVecToWorld(particle_vel)

    @ti.kernel
    def countGrids(self):
        self.occupiedFieldCount[0] = 0
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            if self.grids[xx, yy, zz].debug_counter > 0:
                gridID = ti.atomic_add(self.occupiedFieldCount[0], 1)
                gridcenter = self.gridPosToWorld(float3(xx, yy, zz) + 0.5)
                self.occupiedGridPositions[gridID] = gridcenter
    
    def countAndRenderOccupiedGrids(self, scene:ti.ui.Scene):
        self.countGrids()
        if self.occupiedFieldCount[0] > 0:
            rad:float = 0.5 * self.grid_width
            scene.particles(centers=self.occupiedGridPositions, radius=rad, index_count=self.occupiedFieldCount[0])

particleList=ParticleGroup(4096*4, initcenter=float3(0.0, 2.0, 0.0), inithalfsize=float3(0.25, 0.5, 0.25))
gridField=GridField(dimension, float3([0, 0, 0]), 0.1)

window = ti.ui.Window("Test 3D Grid Simulation", (1920, 1080), vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 1.0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

deltatime = 1.0 / 60.0
while window.running:
    particleList.tick(deltatime, gridField)

    gridField.cache_grids()
    gridField.insertParticles(particleList)
    gridField.solve_grids()
    gridField.portBackToParticles(particleList)

    camera.position(4, 5, 8)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light([0.5, 0.5, 0.5])
    gridField.draw_field_range(scene)
    #gridField.countAndRenderOccupiedGrids(scene)
    particleList.render(scene)
    frameContext.next_frame()

    canvas.scene(scene)
    window.show()