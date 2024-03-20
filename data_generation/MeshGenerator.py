import gmsh
import math
import numpy as np
import random
import meshio
import argparse



def compute_centroid(points):
    centroid = [sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)]
    return centroid

def polar_angle(point,centroid):
    x, y = point
    angle = math.atan2(y - centroid[1], x - centroid[0])
    return angle

def pair_neighboring_elements(numbers):
    paired_list = []
    length = len(numbers)

    for i in range(length):
        pair = [numbers[i], numbers[(i + 1) % length]]
        paired_list.append(pair)

    return paired_list

def sort_points(points):
    centroid = compute_centroid(points)
    p_keys = [-polar_angle(p,centroid) for p in points]
    sorted_indices = list(np.argsort(p_keys))
    points = [points[s] for s in sorted_indices]

    return points

def rand_points(num_points,max_bound):

    polar_coords = []
    cartesian_coords = []


    for _ in range(num_points):
        r = random.uniform(0, max_bound)  # Random radial distance
        theta = random.uniform(0, 2 * math.pi)  # Random angle in radians
        polar_coords.append((r, theta))

    for r, theta in polar_coords:
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        cartesian_coords.append((x, y))
        
    return cartesian_coords

def compute_distance(p1,p2):
    d = (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
    return d


def polygon_length(points):
    points = sort_points(points)

    d = [compute_distance(points[i],points[i+1]) for i in range(len(points)-1)]

    d.append(compute_distance(points[-1],points[0]))

    return d


def pair_neighboring_elements_internal(line_type):
    paired_list = []

    point_tag = 5

    for lt in line_type[:-1]:
        if lt == 1:
            paired_list.append([point_tag,point_tag+1])
            
            point_tag += 1

        elif lt == 2:
            paired_list.append([point_tag,point_tag+1,point_tag+2])
            point_tag += 2

    if line_type[-1] == 1:
        paired_list.append([point_tag,5])
    elif line_type[-1] == 2:
        paired_list.append([point_tag,point_tag+1,5])

    
    return paired_list


def calculate_points(total_length, distance_between_points):
    if distance_between_points <= 0:
        raise ValueError("Distance between points must be greater than 0")
    
    if total_length <= 0:
        raise ValueError("Total length must be greater than 0")
    
    num_points = total_length / distance_between_points
    return int(num_points) + 1


def generate_circle_points(n, radius, center=(0, 0)):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    coordinates = np.column_stack((x, y))
    return coordinates


def add_circle(x_center,y_center,n,radius):
    cp = generate_circle_points(n,radius,(x_center,y_center))
    center_point = gmsh.model.geo.addPoint(x_center,y_center,0)
    cp_tags = []
    for p in cp:
        x,y = p
        cp_tags.append(gmsh.model.geo.addPoint(x,y,0))

    line_tags = []
    for i in range(len(cp_tags)):
        line_tags.append(gmsh.model.geo.addCircleArc(cp_tags[i%len(cp_tags)],center_point,cp_tags[(i+1)%len(cp_tags)]))

    curve_loop_tag = gmsh.model.geo.add_curve_loop(line_tags)

    return curve_loop_tag


def add_rand_partial_circle(x_center,y_center,n,radius):
    cp = generate_circle_points(n,radius,(x_center,y_center))
    center_point = gmsh.model.geo.addPoint(x_center,y_center,0)
    cp_tags = []

    print(len(cp))
    section_length = random.randint(20,len(cp)-1)
    section_start = random.randint(0,len(cp))
    cp = circular_slice(cp,section_start,section_length)

    for p in cp:
        x,y = p
        cp_tags.append(gmsh.model.geo.addPoint(x,y,0))

    line_tags = []
    for i in range(len(cp_tags)-1):
        line_tags.append(gmsh.model.geo.addCircleArc(cp_tags[i],center_point,cp_tags[(i+1)]))
    
    line_tags.append(gmsh.model.geo.addLine(cp_tags[-1],cp_tags[0]))

    curve_loop_tag = gmsh.model.geo.add_curve_loop(line_tags)

    return curve_loop_tag


def circular_slice(lst, start, length):
    n = len(lst)
    end = (start + length) % n
    if end >= start:
        return np.array(lst[start:end])
    else:
        return np.concatenate((lst[start:], lst[:end]))


def circles_overlap(circle1,circle2):
    x1,y1,r1 = circle1
    x2,y2,r2 = circle2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance <= (r1 + r2 + 0.01)

def triangle_area(x1, y1, x2, y2, x3, y3):
    # Shoelace Formula
    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area

def generate_mesh(output_filename,n_points):

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add('model')
    points = [(-0.5,-0.5),(-0.5,0.5),(1.0,-0.5),(1.0,0.5)]

    curve_loops = []

    points = sort_points(points)

    print(points)

    bounding_box_tags = []

    for i,p in enumerate(points):
        x = p[0]
        y = p[1]
        bounding_box_tags.append(gmsh.model.geo.add_point(x,y,0,tag=i+1))
    

    line_pairs = pair_neighboring_elements([i+1 for i in range(len(points))])
    
    lines = []
    #add lines
    for i,l in enumerate(line_pairs):
        lines.append(gmsh.model.geo.addLine(l[0], l[1]))


    curve_loops.append(gmsh.model.geo.add_curve_loop(lines))

    circles = []

    circle_curve_loops = []

    while len(circles) < n_points:
        
        
        current_circle = (random.uniform(-0.4,0.4),random.uniform(-0.4,0.4),random.uniform(0.02,0.1))
        n = 3

        circle_overlap_flag = False
        
        for c in circles:
            
            if circles_overlap(c,current_circle):
                circle_overlap_flag = True
                break

        if circle_overlap_flag:
            continue

        

        
        
        circle_curve_loop = add_circle(current_circle[0],current_circle[1],n,current_circle[2])
        curve_loops.append(circle_curve_loop)
        circle_curve_loops.append(circle_curve_loop)
        
        circles.append(current_circle)

        print(len(circles),n_points)




    surface = []
    surface.append(gmsh.model.geo.addPlaneSurface(curve_loops))


    gmsh.model.geo.synchronize()

    s = gmsh.model.geo.extrude([(2,1)],0,0,0.1,[1],[1],recombine=True)


    gmsh.model.addPhysicalGroup(2,[s[0][1],1],name = "FrontBackPlane")

    gmsh.model.addPhysicalGroup(2,[s[3][1]],name="outflow")    

    gmsh.model.addPhysicalGroup(2,[s[5][1]],name="inflow")

    gmsh.model.addPhysicalGroup(2,[s[2][1],s[4][1]],name="sidewalls")

    gmsh.model.addPhysicalGroup(2,[s[i][1] for i in range(6,len(s))],name="wall")
    gmsh.model.addPhysicalGroup(3,[1],name="internal")
    


    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [s[i][1] for i in range(6,len(s))])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    #res_min = 0.003
    res_min = random.uniform(0.003,0.004)

    gmsh.model.mesh.field.add("Threshold",2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", res_min)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", res_min*2.9)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write(output_filename)

    msh = meshio.read(output_filename)
    triangles = msh.cells_dict['triangle'][(msh.points[msh.cells_dict['triangle']][:,:,-1] == 0)[:,0]]
    mesh_points = msh.points

    t = mesh_points[triangles]
    x1 = t[:,0,0]
    y1 = t[:,0,1]
    x2 = t[:,1,0]
    y2 = t[:,1,1]
    x3 = t[:,2,0]
    y3 = t[:,2,1]

    area = triangle_area(x1,y1,x2,y2,x3,y3)

    print(f"mesh area ratio: {area.max()/area.min()}")
    print(mesh_points.shape)



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('output_filename', type=str, help='output filename of mesh')
    parser.add_argument('n_objects', type=int, help='number of objects in case')

    args = parser.parse_args()
    
    output_filename = args.output_filename
    n_points = args.n_objects

    generate_mesh(output_filename,n_points)