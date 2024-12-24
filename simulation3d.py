import taichi as ti
import taichi.math as tm
import numpy as np
import math
ti.init(arch=ti.cuda)

int3 = ti.types.vector(3, dtype=ti.i32)
float3 = ti.types.vector(3, float)

dimension = int3([64, 64, 64])
gravity = float3([0, -9.8, 0])

unified_initialization=True

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

@ti.func
def vector_is_nan(in_vec:float3):
    nan_val = tm.isnan(in_vec)
    return (in_vec[0] + in_vec[1] + in_vec[2]) > 0

@ti.func
def number_is_nan(i_number:ti.template()):
    return tm.isnan(i_number)>0

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
            mask_positive = 1.0
            mask_negative = 1.0
            newman_bound = True
            for rk in range(self.filter_rank_num):
                center_val = in_grid[rk, center_pos]
                center_weight = self.filter_kernel[rk, 0]
                combined_val = center_val * center_weight
                combined_weight = center_weight
                for filter_id in range(1, self.filter_half_width):
                    if self.predict_boundary(center_pos, pos_positive, dir_positive, dim):
                        if newman_bound:
                            dir_positive = -dir_positive
                        else:
                            dir_positive = 0
                            mask_positive = 0.0
                    if self.predict_boundary(center_pos, pos_negative, dir_negative, dim):
                        if newman_bound:
                            dir_negative = -dir_negative
                        else:
                            dir_negative = 0
                            mask_negative = 0.0
                    pos_positive += dir_positive
                    pos_negative += dir_negative
                    val_positive = in_grid[rk, center_pos + pos_positive * direction] * mask_positive
                    weight_positive = self.filter_kernel[rk, filter_id]
                    val_negative = in_grid[rk, center_pos + pos_negative * direction] * mask_negative
                    weight_negative = self.filter_kernel[rk, filter_id]
                    combined_val += val_positive * weight_positive + val_negative * weight_negative
                    combined_weight += weight_positive + weight_negative

                # if tm.isnan(combined_weight)>0:
                #     print("nan")

                # if combined_weight != 0:
                #     combined_val /= combined_weight
                # else:
                #     combined_val = 0.0
                #nan_test = tm.isnan(result_div)

                # if tm.isnan(combined_val)>0:
                #     print("nan1")
                #     combined_val = 0.0
                out_grid[rk, center_pos] = combined_val

    @ti.kernel
    def initialize_pressure(self, in_pressure:ti.template(), inout_pressure:ti.template(), dim:int3):
        for xx, yy, zz, in ti.ndrange(dim[0], dim[1], dim[2]):
            ini_pres = in_pressure[xx, yy, zz]
            ini_sign = 1.0
            # if unified_initialization:
            #     ini_sign = -1.0
            for rk in range(self.filter_rank_num):
                inout_pressure[rk, int3(xx, yy, zz)] = ini_pres * ini_sign

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

@ti.func
def lerp(a:ti.template(), b:ti.template(), alpha:ti.template()):
    return (1.0 - alpha) * a + alpha * b

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
            self.vel *= -1.0
            next_Pos = self.pos + self.vel * delta_time
        self.pos = next_Pos
        if(vector_is_nan(self.vel)):
            self.vel = float3(0, 0, 0)


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
            self.particles[particle_id].initialize(rand_pos, float3([0.0, 0.0, 0]))

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

    @ti.kernel
    def external_forces(self, deltatime:ti.f32, force_center:float3, halfsize:float, accel:float3):
        for particle_id in ti.ndrange(self.particle_count):

            final_accel = gravity
            if tm.length(self.particles[particle_id].pos - force_center) < halfsize:
                final_accel += accel
            self.particles[particle_id].vel += final_accel * deltatime
    
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
        elif self.weight <= 0:
            self.weight = 0.0
            self.vel = float3(0.0, 0.0, 0.0)
        if vector_is_nan(self.vel):
            self.vel = float3(0.0, 0.0, 0.0)

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
    grid_temp_vel=None
    grid_temp_weight=None
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
    def calculate_pressure_vel(self, momentun:bool):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pressure = self.grid_divergence[center_coord]
            right = self.get_neighbour_pressure(center_coord, int3(1, 0, 0), center_pressure)
            left = self.get_neighbour_pressure(center_coord, int3(-1, 0, 0), center_pressure)
            up = self.get_neighbour_pressure(center_coord, int3(0, 1, 0), center_pressure)
            down = self.get_neighbour_pressure(center_coord, int3(0, -1, 0), center_pressure)
            front = self.get_neighbour_pressure(center_coord, int3(0, 0, 1), center_pressure)
            back = self.get_neighbour_pressure(center_coord, int3(0, 0, -1), center_pressure)
            result_vel = float3(right - left, up - down, front - back) / 2.0
            if momentun:
                result_vel /= self.grids[center_coord].weight
            self.grid_pressure_vel_resolved[center_coord] = result_vel

    
    @ti.kernel
    def do_subtract_pressure_vel(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            self.grids[xx, yy, zz].vel -= self.grid_pressure_vel_resolved[xx, yy, zz]


    def project_pressure(self, momentun:bool):
        self.calculate_pressure_vel(momentun)
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

                # if(vector_is_nan(combined_del_vel) or number_is_nan(combined_weight)):
                #     print("input Nan")

                if combined_weight > 0:
                    combined_del_vel /= combined_weight
                    #combined_vel /= combined_weight

                #print("[",frameContext.get_frame(),"]","del_vel:", combined_del_vel)
                
                particle_vel = self.worldVecToGrid(particleGroup.particles[particle_id].vel)
                newVelFlip = particle_vel + combined_del_vel
                flip_str = 0.0
                particle_vel = flip_str * newVelFlip + (1.0 - flip_str) * combined_vel
                #print(particle_vel)
                # if vector_is_nan(combined_del_vel):
                #     print("nan!!")

                vel_length = tm.length(particle_vel)
                if vel_length > 500.0:
                    particle_vel *= 500.0 / vel_length

                if vector_is_nan(particle_vel):
                    particle_vel = float3(0, 0, 0)

                # if number_is_nan(particle_vel):
                #     print("nan")
                #print("combined_vel,", combined_vel)
                particleGroup.particles[particle_id].vel = self.gridVecToWorld(particle_vel)

    @ti.kernel
    def countGrids(self):
        self.occupiedFieldCount[0] = 0
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            if self.grids[xx, yy, zz].weight > 0.0:
                gridID = ti.atomic_add(self.occupiedFieldCount[0], 1)
                gridcenter = self.gridPosToWorld(float3(xx, yy, zz) + 0.5)
                self.occupiedGridPositions[gridID] = gridcenter

    @ti.func
    def get_neighbour_vel(self, center_coord:int3, bias:int3, center_vel:float3, momentum:bool):
        neighbour_coord = center_coord + bias
        result_vel = center_vel * -bias
        if valid_range(neighbour_coord, self.grid_dimension):
            result_vel = self.grids[neighbour_coord].vel
            if momentum:
                result_vel *= self.grids[neighbour_coord].weight
        return result_vel

    @ti.kernel
    def divergence(self, momentum:bool):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            grid_vel = self.grids[center_coord].vel
            if(momentum):
                grid_vel *= self.grids[center_coord].weight
            right = self.get_neighbour_vel(center_coord, int3(1, 0, 0), grid_vel, momentum)[0]
            left = self.get_neighbour_vel(center_coord, int3(-1, 0, 0), grid_vel, momentum)[0]
            up = self.get_neighbour_vel(center_coord, int3(0, 1, 0), grid_vel, momentum)[1]
            down = self.get_neighbour_vel(center_coord, int3(0, -1, 0), grid_vel, momentum)[1]
            front = self.get_neighbour_vel(center_coord, int3(0, 0, 1), grid_vel, momentum)[2]
            back = self.get_neighbour_vel(center_coord, int3(0, 0, -1), grid_vel, momentum)[2]
            result_div = (right - left + up - down + front - back) / 6.0
            self.grid_divergence[center_coord] = result_div

    @ti.kernel
    def add_mass(self, center:float3, halfsize:float):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pos = ti.cast(center_coord, ti.f32) + 0.5
            grid_local_force_center=self.worldPosToGrid(center)
            grid_local_halfsize = halfsize / self.grid_width
            if(tm.length(grid_local_force_center - center_pos) < grid_local_halfsize):
                self.grids[center_coord].weight = 1.0


    @ti.kernel
    def external_forces(self, deltatime:ti.f32, force_center:float3, halfsize:float, iniVel:float3):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pos = ti.cast(center_coord, ti.f32) + 0.5
            grid_local_force_center=self.worldPosToGrid(force_center)
            grid_local_halfsize = halfsize / self.grid_width
            # if(tm.length(grid_local_force_center - center_pos) < grid_local_halfsize):
            #     self.grids[center_coord].vel = self.worldVecToGrid(iniVel)
            #     self.grids[center_coord].weight = 1.0
 
            if self.grids[center_coord].weight > 0:
                self.grids[center_coord].vel += self.worldVecToGrid(gravity) * deltatime

    @ti.func
    def sample_velocity(self, sample_coord:float3):
        vel_mask = 0.0
        if self.valid_local_position(sample_coord):
            vel_mask = 1.0
        subtracted_pos = tm.clamp(sample_coord - 0.5, 0.0, ti.cast(self.grid_dimension, ti.f32) - 1.0001)
        base_coord = ti.cast(tm.floor(subtracted_pos), ti.i32)
        fr = tm.fract(subtracted_pos)
        inv_fr = 1.0 - fr
        
        result = float3(0, 0, 0)
        for xx in range(2):
            for yy in range(2):
                for zz in range(2):
                    bias = int3(xx, yy, zz)
                    weight3 = lerp(inv_fr, fr, ti.cast(bias, ti.f32))
                    result += weight3[0] * weight3[1] * weight3[2] * self.grids[base_coord + bias].vel

        # nan_test = tm.isnan(result)
        # if (nan_test[0] + nan_test[1] + nan_test[2]) > 0:
        #     print("nan")
        return result * vel_mask

    
    @ti.kernel
    def semi_laglarian_advect_grids(self, outvels:ti.template(), delta_time:ti.f32):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pos = ti.cast(center_coord, ti.f32) + 0.5
            grid_vel = self.grids[center_coord].vel
            trace_back = center_pos - grid_vel * delta_time
            outvels[center_coord] = self.sample_velocity(trace_back)

    @ti.func
    def overlap_area(self, center1:ti.template(), halfwidth1:float, center2:ti.template(), halfwidth2:float):
        max_min = tm.max(center1 - halfwidth1, center2 - halfwidth2)
        min_max = tm.min(center1 + halfwidth1, center2 + halfwidth2)
        result = 0.0;
        if(min_max[0] > max_min[0] and min_max[1] > max_min[1] and min_max[2] > max_min[2]):
            dist = min_max - max_min
            result = dist[0] * dist[1] * dist[2]
        return result
    
    @ti.kernel
    def cleanup_temp_vel_and_weights(self, outvels:ti.template(), outweights:ti.template()):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            outvels[xx, yy, zz] = float3(0, 0, 0)
            outweights[xx, yy, zz] = 0.0

    @ti.kernel
    def sanitize_grid(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            if vector_is_nan(self.grids[xx, yy, zz].vel):
                self.grids[xx, yy, zz].vel = float3(0.0, 0.0, 0.0)
    
    @ti.kernel
    def grid_forward_advect(self, deltaTime:float, outvels:ti.template(), outweights:ti.template()):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pos = ti.cast(center_coord, ti.f32) + 0.5
            grid_vel = self.grids[center_coord].vel
            gridweight = self.grids[center_coord].weight
            if gridweight > 0.0:
                next_pos = center_pos + deltaTime * grid_vel
                if not self.valid_local_position(next_pos):
                    grid_vel *= -1.0
                    next_pos = center_pos + deltaTime * grid_vel
                center_pos = next_pos
                grid_halfwidth = 0.5
                grid_min = tm.clamp(ti.cast(center_pos - grid_halfwidth, ti.i32), 0, self.grid_dimension - 1)
                grid_max = tm.clamp(ti.cast(center_pos + grid_halfwidth, ti.i32), 0, self.grid_dimension - 1)
                grid_range = grid_max + 1 - grid_min
                if(grid_range[0] > 0 and grid_range[1] > 0 and grid_range[2] > 0):
                    for xi in range(grid_range[0]):
                        for yi in range(grid_range[1]):
                            for zi in range(grid_range[2]):
                                itr_grid = grid_min + int3(xi, yi, zi)
                                itr_grid_center = ti.cast(itr_grid, ti.f32) + 0.5
                                overlapped_weight = self.overlap_area(itr_grid_center, 0.5, center_pos, grid_halfwidth)
                                ti.atomic_add(outvels[itr_grid], overlapped_weight * grid_vel * gridweight)
                                ti.atomic_add(outweights[itr_grid], overlapped_weight * gridweight)
                            
    @ti.kernel
    def validate_vel(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            if vector_is_nan(self.grids[xx, yy, zz].vel):
                self.grids[xx, yy, zz].vel = float3(0.0, 0.0, 0.0)
                print("isNan")

    @ti.kernel
    def validate_temp_vec(self, tmp_vec:ti.template()):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            if vector_is_nan(tmp_vec[xx, yy, zz]):
                tmp_vec[xx, yy, zz] = float3(0.0, 0.0, 0.0)
                print("isNan")

    @ti.kernel
    def apply_velocity(self, inVels:ti.template()):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            self.grids[center_coord].vel = inVels[center_coord]

    @ti.kernel
    def apply_weight(self, inWeights:ti.template()):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            self.grids[center_coord].weight = inWeights[center_coord]

    def advect_semi_laglarian(self, delta_time):
        if self.grid_temp_vel == None:
            self.grid_temp_vel = float3.field(shape=self.grid_dimension)
        self.semi_laglarian_advect_grids(self.grid_temp_vel, delta_time)
        self.apply_velocity(self.grid_temp_vel)

    def advect_grid_forward(self, delta_time):
        if self.grid_temp_weight == None:
            self.grid_temp_weight = ti.field(ti.f32, shape=self.grid_dimension)
        if self.grid_temp_vel == None:
            self.grid_temp_vel = float3.field(shape=self.grid_dimension)
        self.grid_temp_weight.fill(0.0)
        self.grid_temp_vel.fill(float3(0, 0, 0))
        self.sanitize_grid()
        #self.cleanup_temp_vel_and_weights(self.grid_temp_vel, self.grid_temp_weight)
        self.grid_forward_advect(delta_time, self.grid_temp_vel, self.grid_temp_weight)
        #self.validate_temp_vec(self.grid_temp_vel)
        self.apply_weight(self.grid_temp_weight)
        self.apply_velocity(self.grid_temp_vel)
        #self.validate_vel()
        self.normalize_velocity()
        

    def countAndRenderOccupiedGrids(self, scene:ti.ui.Scene):
        self.countGrids()
        if self.occupiedFieldCount[0] > 0:
            rad:float = 0.5 * self.grid_width
            scene.particles(centers=self.occupiedGridPositions, radius=rad, index_count=self.occupiedFieldCount[0])
    @ti.func
    def weight_to_pressure(self, density:float):
        return tm.pow(density, 7.0) - 1.0

    @ti.func
    def pressure_weight(distance:float, h:float):
        w = tm.max(0.0, h - distance)
        return w * w * w

    @ti.func
    def calc_sph_pressure(self, center:int3, neighbour:int3):
        dist_vec = ti.cast(center - neighbour, ti.f32)
        center_density = self.grids[center].weight
        neighbour_density = self.grids[neighbour].weight
        center_pres = self.weight_to_pressure(center_density)
        neighbour_pres = self.weight_to_pressure(neighbour_density)
        pressure_w = self.pressure_weight(tm.length(dist_vec), 1.5)
        return (center_pres / tm.max(0.001, center_density * center_density) + neighbour_pres / tm.max(0.001, neighbour_density * neighbour_density)) * pressure_w * tm.normalize(dist_vec)

    @ti.kernel
    def sph_solve(self):
        for xx, yy, zz in ti.ndrange(self.grid_dimension[0], self.grid_dimension[1], self.grid_dimension[2]):
            center_coord = int3(xx, yy, zz)
            center_pos = ti.cast(center_coord, ti.f32) + 0.5
            grid_vel = self.grids[center_coord].vel
            gridweight = self.grids[center_coord].weight
            if gridweight > 0:
                for xx in range(-1, 2):
                    for yy in range(-1, 2):
                        for zz in range(-1, 2):
                            if not(xx==0 and yy==0 and zz==0):
                                bias=int3(xx, yy, zz)
                                neighbour_coord=center_coord+bias
                                local_weight = self.calc_sph_pressure(center_coord, neighbour_coord)


particleList=ParticleGroup(1024, initcenter=float3(0.0, 0.0, 0.0), inithalfsize=float3(1.0, 1.0, 1.0))
gridField=GridField(dimension, float3([0, 0, 0]), 0.05)

window = ti.ui.Window("Test 3D Grid Simulation", (1920, 1080), vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 1.0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

euler_mode=True
semi_laglarian=False

# if euler_mode:
#     gridField.insertParticles(particleList)
#     gridField.normalize_velocity()


deltatime = 1.0 / 60.0

#gridField.external_forces(deltatime, force_center=float3(0, 0, 0), halfsize=0.2, iniVel=float3(0.0, 0.0, 0.0))
gridField.add_mass(center=float3(0, 0, 0), halfsize=0.5)
use_momentun=False
while window.running:

    if not euler_mode:
        gridField.cache_grids()
        gridField.insertParticles(particleList)
        gridField.normalize_velocity()
    else:
        gridField.external_forces(deltatime, force_center=float3(0, 0, 0), halfsize=0.2, iniVel=float3(4.0, 0.0, 0.0))
    gridField.divergence(use_momentun)
    possionKernel.solve_grid_pressure(gridField)
    gridField.project_pressure(use_momentun)
    if euler_mode:
        if semi_laglarian:
            gridField.advect_semi_laglarian(deltatime)
        else:
            gridField.advect_grid_forward(deltatime)

    #if not euler_mode:
    gridField.portBackToParticles(particleList)
    #particleList.external_forces(deltatime,  force_center=float3(0, 0, 0), halfsize=1.0, accel=float3(4.0, 0.0, 0.0))
    particleList.tick(deltatime, gridField)

    camera.position(4, 5, 8)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light([0.5, 0.5, 0.5])
    gridField.draw_field_range(scene)
    gridField.countAndRenderOccupiedGrids(scene)
    #particleList.render(scene)
    #frameContext.next_frame()

    canvas.scene(scene)
    window.show()