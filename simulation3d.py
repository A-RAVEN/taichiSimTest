import taichi as ti
import taichi.math as tm
import numpy as np
import math
ti.init(arch=ti.cuda)

int3 = ti.types.vector(3, dtype=ti.i32)
float3 = ti.types.vector(3, float)

dimension = int3([128, 128, 128])
gravity = float3([0, -9.8, 0])

filter_data = np.load("single_poisson_D3_INVERSE_STANDARD_dx_1.0_itr_32.npy", allow_pickle=True)[0]
# filter_rank_num = filter_data.shape[0]
# filter_width = filter_data.shape[1]
# filter_half_width = math.ceil(filter_width / 2)
# print("Rank Num:", filter_rank_num)
# print("Filter Width:", filter_width)
# print("Filter Half Width:", filter_half_width)
print("Rank1:", filter_data[0])
# print("Rank2:", filter_data[1])
# print("Rank3:", filter_data[2])
# print("Rank4:", filter_data[3])

@ti.func
def valid_range(pos:ti.template(), dim:ti.template()):
    return (pos[0] > 0 and pos[0] < dim[0])\
        and (pos[1] > 0 and pos[1] < dim[1])\
        and (pos[2] > 0 and pos[2] < dim[2])

@ti.data_oriented
class PossionKernel:
    filter_kernel=None
    filter_rank_num:int
    filter_width:int
    filter_half_width:int
    tmp_pressure_ping=None
    tmp_pressure_pong=None
    def __init__(self, in_filter_data):
        self.filter_rank_num = in_filter_data.shape[0]
        self.filter_width = in_filter_data.shape[1]
        self.filter_half_width = math.ceil(self.filter_width / 2)
        print("Kernel Info:")
        print("\tRank Num:", self.filter_rank_num)
        print("\tFilter Width:", self.filter_width)
        print("\tFilter Half Width:", self.filter_half_width)
        self.filter_kernel = ti.field(ti.f32, shape=(self.filter_rank_num, self.filter_half_width))
        for rk in range(self.filter_rank_num):
            for kid in range(self.filter_half_width):
                self.filter_kernel[rk, kid] = in_filter_data[rk][self.filter_half_width + kid - 1]
        print("\tFIlterMatrix:", self.filter_kernel)

    @ti.func
    def predict_boundary(self, in_center_pos:int3, in_pos:int, in_dir:int, in_dim:int3):
        next_pos = in_center_pos + in_dim * (in_pos + in_dir)
        return valid_range(next_pos, in_dim)


    @ti.kernel
    def solve_dir(self, direction:int3, in_grid:ti.template(), out_grid:ti.template(), dim:int3):
        for xx, yy, zz, in ti.ndrange(dim[0], dim[1], dim[2]):
            center_pos = int3([xx, yy, zz])
            dir_positive = 1
            dir_negative = -1
            pos_positive = 0
            pos_negative = 0
            for rk in range(self.filter_rank_num):
                center_val = in_grid[rk, center_pos]
                center_weight = self.filter_kernel[rk, 0]
                combined_val = center_val * center_weight
                combined_weight = center_weight
                for filter_id in range(1, self.filter_half_width):
                    if self.predict_boundary(center_pos, pos_positive, dir_positive, dim):
                        dir_positive = -dir_positive
                    if self.predict_boundary(center_pos, pos_negative, dir_negative, dim):
                        dir_negative = -dir_negative
                    pos_positive += dir_positive
                    pos_negative += dir_negative
                    val_positive = in_grid[rk, center_pos + pos_positive * direction]
                    weight_positive = self.filter_kernel[rk, abs(pos_positive)]
                    val_negative = in_grid[rk, center_pos + pos_negative * direction]
                    weight_negative = self.filter_kernel[rk, abs(pos_negative)]
                    combined_val += val_positive * weight_positive + val_negative * weight_negative
                    combined_weight += weight_positive + weight_negative

                if tm.isnan(combined_weight)>0:
                    print("nan")

                if combined_weight != 0:
                    combined_val /= combined_weight
                else:
                    combined_val = 0.0
                #nan_test = tm.isnan(result_div)

                if tm.isnan(combined_val)>0:
                    combined_val = 0.0
                out_grid[rk, center_pos] = combined_val

    @ti.kernel
    def initialize_pressure(self, in_pressure:ti.template(), inout_pressure:ti.template(), dim:int3):
        for xx, yy, zz, in ti.ndrange(dim[0], dim[1], dim[2]):
            ini_pres = in_pressure[xx, yy, zz]
            for rk in range(self.filter_rank_num):
                inout_pressure[rk, int3(xx, yy, zz)] = ini_pres

    @ti.kernel
    def resolve_back_pressure(self, in_pressure_multi_rank:ti.template(), out_pressure:ti.template(), dim:int3):
        for xx, yy, zz, in ti.ndrange(dim[0], dim[1], dim[2]):
            combined_pressure = 0.0
            for rk in range(self.filter_rank_num):
                combined_pressure +=in_pressure_multi_rank[rk, xx, yy, zz]

            #print(combined_pressure)
            out_pressure[xx, yy, zz] = combined_pressure

    def solve_grid_pressure(self, ingridField):
        if self.tmp_pressure_ping==None:
            self.tmp_pressure_ping = ti.field(float, shape=(self.filter_rank_num, ingridField.grid_dimension[0], ingridField.grid_dimension[1], ingridField.grid_dimension[2]))
        if self.tmp_pressure_pong==None:
            self.tmp_pressure_pong = ti.field(float, shape=(self.filter_rank_num, ingridField.grid_dimension[0], ingridField.grid_dimension[1], ingridField.grid_dimension[2]))
        self.initialize_pressure(ingridField.grid_divergence, self.tmp_pressure_ping, ingridField.grid_dimension)
        self.solve_dir(int3(1, 0, 0), self.tmp_pressure_ping, self.tmp_pressure_pong, ingridField.grid_dimension)
        self.solve_dir(int3(0, 1, 0), self.tmp_pressure_pong, self.tmp_pressure_ping, ingridField.grid_dimension)
        self.solve_dir(int3(0, 0, 1), self.tmp_pressure_ping, self.tmp_pressure_pong, ingridField.grid_dimension)
        self.resolve_back_pressure(self.tmp_pressure_pong, ingridField.grid_divergence, ingridField.grid_dimension)
            
possionKernel = PossionKernel(filter_data)

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
        self.vel += gravity * delta_time
        #self.vel *= 0.8
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
            self.particles[particle_id].initialize(rand_pos, float3([0, 0, 0]))

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
        if self.weight > 1:
            self.vel /= self.weight
        elif self.weight <= 0:
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
    grid_divergence=None
    grid_pressure_vel_resolved=None
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
        self.grid_divergence = ti.field(float, shape=self.grid_dimension)
        self.grid_pressure_vel_resolved = float3.field(shape=self.grid_dimension)
        self.occupiedGridPositions = float3.field(shape=self.grid_dimension[0] * self.grid_dimension[1] * self.grid_dimension[2])
        self.occupiedFieldCount = ti.field(ti.i32, shape=1)
        self.initialize_grids()

    @ti.kernel
    def cache_grids(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].cache()

    @ti.kernel
    def normalize_velocity(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].normalize_velocity()

    @ti.func
    def get_neighbour_pressure(self, center_coord:int3, bias:int3, center_press:float):
        neighbour_coord = center_coord + bias
        result_press = center_press
        if valid_range(neighbour_coord, self.grid_dimension):
            result_press = self.grid_divergence[neighbour_coord]
        return result_press

    @ti.kernel
    def calculate_pressure_vel(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pressure = self.grid_divergence[center_coord]
            right = self.get_neighbour_pressure(center_coord, int3(1, 0, 0), center_pressure)
            left = self.get_neighbour_pressure(center_coord, int3(-1, 0, 0), center_pressure)
            up = self.get_neighbour_pressure(center_coord, int3(0, 1, 0), center_pressure)
            down = self.get_neighbour_pressure(center_coord, int3(0, -1, 0), center_pressure)
            front = self.get_neighbour_pressure(center_coord, int3(0, 0, 1), center_pressure)
            back = self.get_neighbour_pressure(center_coord, int3(0, 0, -1), center_pressure)
            result_vel = float3(right - left, up - down, front - back) * 0.5
            self.grid_pressure_vel_resolved[center_coord] = result_vel

    
    @ti.kernel
    def do_subtract_pressure_vel(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].vel -= self.grid_pressure_vel_resolved[xx, yy, zz]


    def project_pressure(self):
        self.calculate_pressure_vel()
        self.do_subtract_pressure_vel()

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
                flip_str = 0.0
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
            if self.grids[xx, yy, zz].weight > 0:
                gridID = ti.atomic_add(self.occupiedFieldCount[0], 1)
                gridcenter = self.gridPosToWorld(float3(xx, yy, zz) + 0.5)
                self.occupiedGridPositions[gridID] = gridcenter

    @ti.func
    def get_neighbour_vel(self, center_coord:int3, bias:int3, center_vel:float3):
        neighbour_coord = center_coord + bias
        result_vel = float3(0, 0, 0)#center_vel * -bias
        if valid_range(neighbour_coord, self.grid_dimension):
            result_vel = self.grids[neighbour_coord].vel
        return result_vel

    @ti.kernel
    def divergence(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            grid_vel = self.grids[center_coord].vel
            right = self.get_neighbour_vel(center_coord, int3(1, 0, 0), grid_vel)[0]
            left = self.get_neighbour_vel(center_coord, int3(-1, 0, 0), grid_vel)[0]
            up = self.get_neighbour_vel(center_coord, int3(0, 1, 0), grid_vel)[1]
            down = self.get_neighbour_vel(center_coord, int3(0, -1, 0), grid_vel)[1]
            front = self.get_neighbour_vel(center_coord, int3(0, 0, 1), grid_vel)[2]
            back = self.get_neighbour_vel(center_coord, int3(0, 0, -1), grid_vel)[2]
            result_div = (right - left + up - down + front - back) / 6.0
            # if result_div != 0:
            #     print(result_div)

            self.grid_divergence[center_coord] = result_div
    
    def countAndRenderOccupiedGrids(self, scene:ti.ui.Scene):
        self.countGrids()
        if self.occupiedFieldCount[0] > 0:
            rad:float = 0.5 * self.grid_width
            scene.particles(centers=self.occupiedGridPositions, radius=rad, index_count=self.occupiedFieldCount[0])

particleList=ParticleGroup(4096*4, initcenter=float3(0.0, 2.0, 0.0), inithalfsize=float3(0.25, 0.5, 0.25))
gridField=GridField(dimension, float3([0, 0, 0]), 0.05)

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
    gridField.normalize_velocity()
    gridField.divergence()
    possionKernel.solve_grid_pressure(gridField)
    gridField.project_pressure()
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