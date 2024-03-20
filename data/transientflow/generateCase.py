import os
import argparse
import shutil
import random
import time
from tqdm import tqdm
import pickle
from fluidfoam import readscalar
from fluidfoam import readmesh
from fluidfoam import readvector
from sklearn.cluster import DBSCAN
import meshio
from shapely.geometry import Point,Polygon

import torch
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import io
from PIL import Image
from multiprocessing import Process

from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedParameterFile import ParsedBoundaryDict

from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Execution.ParallelExecution import LAMMachine

from MeshGenerator import generate_mesh,sort_points



def generate_object_mask(sol_dir,x_res,y_res):

    msh = meshio.read(sol_dir+"/mesh.msh")
    
    eps = 0.0075
    min_samples = 2
    dbscan = DBSCAN(eps=eps,min_samples=min_samples)
    
    ngridx = x_res
    ngridy = y_res

    msh.points[:,1].max()
    
    xinterpmin = msh.points[:,0].min()
    
    xinterpmax = msh.points[:,0].max()

    
    yinterpmin = msh.points[:,1].min()
    
    yinterpmax = msh.points[:,1].max()

    xi = np.linspace(xinterpmin,xinterpmax,ngridx)
    yi = np.linspace(yinterpmin,yinterpmax,ngridy)
    xinterp,yinterp = np.meshgrid(xi,yi)


    wall_points = msh.points[msh.cells_dict['quad'][msh.cell_sets_dict['wall']['quad']]]
    wp_corrected = wall_points[(wall_points[:,:,2]==0)][:,:2][1:-1:2]
    wp_corrected = [(p[0],p[1]) for p in wp_corrected]
    wp_corrected = torch.tensor(wp_corrected)

    clusters = dbscan.fit_predict(wp_corrected)

    if len(set(clusters)) < 1:
        return None

    p_clusters = []
    for cluster_id in set(clusters):
        cluster_points = torch.cat([wp_corrected[clusters==cluster_id],wp_corrected[clusters==cluster_id][0].view(1,2)])
        p_clusters.append(cluster_points)

    polygon_list = [Polygon(sort_points([(p[0].item(),p[1].item()) for p in p_clusters[i]])) for i in range(len(p_clusters))]

    interp = torch.tensor(np.stack((xinterp,yinterp),axis=2)).flatten(0,1)

    object_mask = []
    for p in tqdm(interp):
        mask_value = 0
        for polygon in polygon_list:
            mask_value += polygon.contains(Point(p))
            
        object_mask.append(mask_value)


    object_mask = torch.tensor(object_mask).view(ngridy,ngridx).flip(0)
    return object_mask,len(set(clusters))




def readU(arg):
    i,dest = arg
    return torch.tensor(readvector(dest,str(i),'U'))

def readp(arg):
    i,dest = arg
    return torch.tensor(readscalar(dest,str(i),'p'))

def readPhi(arg):
    i,dest = arg
    return torch.tensor(readscalar(dest,str(i),'phi'))

plot_height = 5.0
def scatter_plot(arg):
    i,x,y,v,triangles,mesh_points = arg
    fig = plt.figure(figsize=(plot_height*2.8, plot_height), dpi=100)
    plt.tripcolor(mesh_points[:,0],mesh_points[:,1],triangles,v[i],alpha=1.0,shading='flat', antialiased=True, linewidth=0.72,edgecolors='face')
    img_buf = io.BytesIO()
    fig.savefig(img_buf,format='png')
    plt.close(fig)
    #print(i)
    return Image.open(img_buf)



def prepareCase(src,dest,n_points,velocity,n_cores,x_res,y_res):

    if os.path.exists(dest) and os.path.isdir(dest):
        shutil.rmtree(dest)

    destination = shutil.copytree(src,dest)

    

    while True:
        process = Process(target=generate_mesh,args=(dest+"mesh.msh",n_points))

        try:
            process.start()
            process.join()
        except Exception as e:
            print("retry mesh generation")
        
        finally:
            process.terminate()
        

        object_mask,n_detected_objects = generate_object_mask(dest,x_res,y_res)

        if process.exitcode == 0 and object_mask is not None and n_points == n_detected_objects:
            print("Mesh generated")
            break
        else:
            print("retry mesh generation")
            print("objects (expected vs detected):",n_points,n_detected_objects)
            time.sleep(3)

    runner = BasicRunner(argv=["gmshToFoam","-case",dest,dest+"mesh.msh"],logname="logifle",noLog=True)

    runner.start()

    f = ParsedBoundaryDict(dest+"constant/polyMesh/boundary")
    f['FrontBackPlane']['type'] = 'empty'
    f.writeFile()



    f = ParsedParameterFile(dest+"0/U")
    f['internalField'] = 'uniform ('+str(velocity)+' 0 0)'
    f.writeFile()

    f = ParsedParameterFile(dest+"system/decomposeParDict")
    f['numberOfSubdomains'] = n_cores
    f.writeFile()

    runner = BasicRunner(argv=["decomposePar","-case",dest],logname="logifle",noLog=True)

    runner.start()

    return object_mask


def get_current_case(dest):
    current_case_number = -1
    for d in os.listdir(dest):
        #print(d)
        d = d.split('_')
        if d[0] == 'case':
            if int(d[1]) > current_case_number:
                current_case_number = int(d[1])
    return current_case_number+1


def get_current_case_old(dest):
    current_case_number = -1
    iterator = 1
    for d in os.listdir(dest):
        #print(d)
        d = d.split('_')
        if d[0] == 'case':
            if int(d[1]) > current_case_number:
                current_case_number = int(d[1])
    return current_case_number+1



def get_current_case(parent_directory):
    

    # Get a list of all directories in the parent directory
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

    #check if empty
    if not any(directories):
        return 0

    # Extract numerical parts from directory names and convert to integers
    existing_numbers = [int(d.split('_')[1]) for d in directories if d.startswith("case_") and d[5:].isdigit()]

    # Find the lowest missing directory number
    lowest_missing_number = None
    for i in range(1, max(existing_numbers) + 2):
        if i not in existing_numbers:
            lowest_missing_number = i
            break

    return lowest_missing_number


def find_first_available_line(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the first empty line
    empty_line_number = next((i + 1 for i, line in enumerate(lines) if not line.strip()), None)

    if empty_line_number is not None:
        print(f"Found empty line at line {empty_line_number}")
    else:
        # If no empty line is found, create one at the end of the file
        empty_line_number = len(lines) + 1
        lines.append('\n')

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Created empty line at line {empty_line_number}")

    return empty_line_number



def write_status_report(file_path, line_number, new_content):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Check if the specified line number is valid
    if 1 <= line_number <= len(lines):
        # Modify the specific line
        lines[line_number - 1] = new_content + '\n'  # Adding '\n' to maintain proper line endings

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print(f"Content written to line {line_number} in {file_path}")
    else:
        print(f"Invalid line number: {line_number}")




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('n_objects', type=int, help='maximum number of circles/partial circles the case should have (min is 1)')
    parser.add_argument('n_cases', type=int, help='number of cases to be run')
    parser.add_argument('n_cores',type = int, help='number of CPU-cores to use for computation')
    parser.add_argument('empty_case',type=str, help='the empty openfoam case directory')
    parser.add_argument('dest', type=str, help='target directory for the OpenFOAM Cases')
    parser.add_argument('working_dir',type=str, help='working directory for OpenFoam Simulation')

    args = parser.parse_args()

    num_points = args.n_objects
    assert num_points>0, "n_objects < 1"

    x_res = 384
    y_res = 256
    

    n_cores = args.n_cores
    assert n_cores>1, "at least two core should be used"

    #save_raw = args.save_raw
    #assert save_raw == 0 or save_raw == 1 , "save_raw is either 0 or 1"
    save_raw = 1

    mpiInformation = LAMMachine(nr=n_cores)

    n_cases = args.n_cases
    src = args.empty_case
    dest = args.dest
    work_dir = args.working_dir

    dest = os.path.join(dest, '')
    work_dir = os.path.join(work_dir, '')

    current_case_number = get_current_case(dest)


    delta_t = [0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001,0.00005,0.000025,0.00001]
    delta_t_index = 0

    

    print("Working directory is: " + work_dir)
    print("Cases are written to: " + dest)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    while current_case_number <= n_cases:
        
        
        
        print("current case: ",str(current_case_number))
        

        n_points = random.randint(1,num_points)
        
        velocity = random.uniform(0.01,0.06)
        

        nr_time_steps = 0

        
        
        crash_counter = 0
        object_mask = prepareCase(src,work_dir,n_points,velocity,n_cores,x_res,y_res)

        msh = meshio.read(work_dir+"/mesh.msh")
        triangles = msh.cells_dict['triangle'][(msh.points[msh.cells_dict['triangle']][:,:,-1] == 0)[:,0]]
        mesh_points = msh.points

        time.sleep(5)
        
        delta_t_index = 0
        f = ParsedParameterFile(work_dir+"system/controlDict")
        max_time_steps = f['endTime']
        f['deltaT'] = delta_t[delta_t_index]
        f.writeFile()

        try:
            while nr_time_steps < max_time_steps:
                
                
                delta_t_index += 1



                if crash_counter > 0:
                    f = ParsedParameterFile(work_dir+"system/controlDict")
                    f['deltaT'] = delta_t[delta_t_index]
                    f.writeFile()

                runner = BasicRunner(argv=["pisoFoam","-case",work_dir],logname="logifle",noLog=True,lam=mpiInformation)
                run_information = runner.start()
                nr_time_steps = run_information['time']
                crash_counter += 1

        except IndexError:
            print("List out of bound, restarting outer loop")
            continue
        
        

        runner = BasicRunner(argv=["redistributePar","-reconstruct","-case",work_dir],logname="logifle",noLog=True,lam=mpiInformation)
        runner.start()

        
        current_case_number = get_current_case(dest)
        solution_dir = dest+"/case_"+str(current_case_number)+"/"
        

        for tries in range(10):
            try:
                os.mkdir(solution_dir)
            except OSError as error:
                print("case already claimed by other process, retry")
                current_case_number = get_current_case(dest)
                solution_dir = dest+"/case_"+str(current_case_number)+"/"
            else:
                print("case available, claiming...")
                break

        os.remove(work_dir+"PyFoamState.CurrentTime")
        os.remove(work_dir+"PyFoamState.LastOutputSeen")
        os.remove(work_dir+"PyFoamState.StartedAt")
        os.remove(work_dir+"PyFoamState.TheState")
        #os.remove(work_dir+"WorkingDirectory.foam")
        for item in os.listdir(work_dir):
            if item.endswith(".foam"):
                os.remove(os.path.join(work_dir,item))



        # convert OpenFOAM to interpolated image
        #os.system(f"python openfoam_to_image.py --src {work_dir} --dst {solution_dir} --grid_height {y_res} --grid_width {x_res}")

        #object_mask = generate_object_mask(work_dir,x_res,y_res)
        torch.save(object_mask,solution_dir+"object_mask.th")



        

        with ParsedParameterFile(work_dir+"0/U") as f:
            initial_velocity = float(str(f['internalField']).split('(')[1].split(" ")[0])

        simulation_description = {"initial_velocity": initial_velocity, "n_objects": n_points}

        

        with open(solution_dir+"simulation_description.pkl",'wb') as handle:
            pickle.dump(simulation_description, handle)


        #shutil.make_archive(solution_dir+"mesh",dest+"mesh.msh",format='bztar')
        shutil.make_archive(base_name=solution_dir+"mesh",
                            format='bztar',
                            root_dir=work_dir,
                            base_dir="mesh.msh")

        

        x,y,z = readmesh(work_dir)
        
        pool_obj = multiprocessing.Pool()
        
        U  = pool_obj.map(readU,[(i,work_dir) for i in range(1,max_time_steps+1)])
        p  = pool_obj.map(readp,[(i,work_dir) for i in range(1,max_time_steps+1)])
        

        pool_obj.close()

        U_stacked = torch.stack(U)
        
        x = torch.tensor(x)
        y = torch.tensor(y)
        v = torch.sqrt(U_stacked[:,0,:]**2+U_stacked[:,1,:]**2+U_stacked[:,2,:]**2)

        
        if save_raw == 1:
            
            for i in range(len(U)):
                local_U = U[i].view(3,-1)[:2]
                local_p = p[i].view(1,-1)
                torch.save(torch.cat([local_U,local_p],dim=0),solution_dir+('{:0>8}'.format(str(i)))+"_mesh.th")
            torch.save(x,solution_dir+"x.th")
            torch.save(y,solution_dir+"y.th")

        shutil.rmtree(work_dir)
        pool_obj = multiprocessing.Pool()
        img_list = pool_obj.map(scatter_plot,[(i,x,y,v,triangles,mesh_points) for i in range(U_stacked.shape[0])])
        
        pool_obj.close()

        for i in range(len(img_list)):
            img_list[i]._min_frame = 0
            img_list[i].n_frames = 1
            img_list[i]._PngImageFile__frame = 0

        img_list[0].save(solution_dir+'U.gif',format='GIF',append_images=img_list[1:],save_all=True,duration=50,loop=0)

        print("solution directory: ", solution_dir)

        
        current_case_number = get_current_case(dest)

        time.sleep(5)

if __name__=="__main__":
    main()
