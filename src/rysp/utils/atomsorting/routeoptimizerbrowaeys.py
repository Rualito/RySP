import numpy as np
import scipy.optimize as sci
import matplotlib.pyplot as plt
import time 
import numba as nb
import ot

def create_template(dim_in,low):
    """
    Creates a square template of dim_in x dim_in spots with the left bottom
    corner at [low,low]

    Parameters
    ----------
    dim_in : [int,int]
        dimensions of the grid
    low : [int,int]
        left corner located at low

    Returns
    -------
    template: np.array of ints with dimension dim_in**2 x 2 
        array with all the locations of the template

    """
    template=np.zeros([dim_in[0]*dim_in[1],2])
    for k in range(dim_in[0]):
        for l in range(dim_in[1]):
            template[dim_in[1]*k+l,:]=[int(low[0]+k),int(low[1]+l)]
    template=np.int64(template)
    return template

def create_grid(dim_out):
    """
    creates a grid of possible spots as dim_out x dim_out

    Parameters
    ----------
    dim_out : [int,int]
        dimension of the square grid

    Returns
    -------
    grid: np.array of ints with dimension dim_out x 2 
        array with all the locations of the grid

    """
    grid=np.zeros([dim_out[0]*dim_out[1],2])
    for k in range(dim_out[0]):
        for l in range(dim_out[1]):
            grid[dim_out[1]*k+l,:]=[k,l]
    grid=np.int64(grid)
    return grid

class RouteOptimizerBrowaeys:
    
    def __init__(self,atoms,template,grid,R):
        self.atoms=atoms
        self.particles=np.zeros([len(atoms),2])
        self.R=R
        for k in range(len(atoms)):
            self.particles[k,:]=[int(atoms[k].pos[0]/self.R),int(atoms[k].pos[1]/self.R)]
        self.template=template
        self.grid=grid
        self.num_particle=len(self.particles)
        self.num_template=len(self.template)
        if self.num_particle<self.num_template:
            print("Too few particles to fill template")
            return
        else:
            self.num_dummies=self.num_particle-self.num_template
        self.cost_matrix=np.zeros([self.num_particle,self.num_particle],np.float64)
        self.assignment=np.zeros([self.num_particle,2])
        self.inverse_assignment=np.zeros([self.num_particle,2])
        self.executed_moves=np.zeros([0,2])
        self.AOM_path=np.zeros([0,2])
        self.order_of_execution=np.arange(0, self.num_template)
        self.AOM_path_new_pos=np.zeros([0,1])
        self.particles_init=np.array(self.particles)
        self.particle_alive=np.zeros([self.num_particle,1])+1
        
        self.succes_move_total=1
        self.succes_move_grid_1=0.95
        self.time_pick_up_drop_off=600
        self.time_single_move=50
        self.individual_atom_lifetime=20*1e6
        
        #print("Initialized Browaeys Optimizer")  
        
    def total_time(self):
        """
        Calculates the time it takes to execute the AOM path

        Returns
        -------
        float
            total time to perform AOM path

        """
        time_pick_up=np.sum(self.AOM_path_new_pos)*self.time_pick_up_drop_off*2
        time_move=(len(self.AOM_path)-np.sum(self.AOM_path_new_pos))*self.time_single_move
        return time_pick_up+time_move
    
    def average_lifetime(self):
        """
        calculates the average lifetime of a template array

        Returns
        -------
        float
            total lifetime of num_template atoms

        """
        return self.individual_atom_lifetime/self.num_template
        
    def update_particle(self,particle_num,new_loc):
        if new_loc in self.particles.tolist():
            print("Already particle in spot: "+str(new_loc))
        else:
            self.particles[particle_num,:]=new_loc
            
    def number_particles_in_block(self,k,l):
        """
        Calculates how many particles are in the block between
        particle l and template k 

        Parameters
        ----------
        k : int
            template index
        l : int 
            particle index

        Returns
        -------
        num_particles : int
            number of particles in the box

        """
        template=self.template[k,:]
        particle=self.particles[l,:]
        x1=min(template[0],particle[0])
        x2=max(template[0],particle[0])
        y1=min(template[1],particle[1])
        y2=max(template[1],particle[1])
        num_particles=0
        for k in range(x1,x2+1):
            for l in range(y1,y2+1):
                num_particles+=np.sum((self.particles==[k,l]).all(1))
        return num_particles
                
    
    def calc_distance(self,p,k,l):
        """
        calculates the distance between particle l and template k
        in l^p

        Parameters
        ----------
        p : float
            power of distance
        k : int
            template index
        l : int
            particle index

        Returns
        -------
        float
            l^p distance between particle l and template k

        """
        return ((np.abs(self.particles[l,0]-self.template[k,0]))**p+
                (np.abs(self.particles[l,1]-self.template[k,1]))**p)**(1/p)
    
    def calc_distance_point_line(self,x0,y0,x1,y1,x2,y2):
        """
        Calculates the distance between point (x0,y0) and the 
        line going through (x1,y1) and (x2,y2)

        Parameters
        ----------
        x0 : float
            x-coordinate of point
        y0 : float
            y-coordinate of point
        x1 : float
            x-coordinate of line
        y1 : float
            y-coordinate of line
        x2 : float
            x-coordinate of line
        y2 : float
            y-coordinate of line

        Returns
        -------
        float
            distance between point and line

        """
        nom=np.abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
        denom=np.sqrt((x1-x2)**2+(y1-y2)**2)
        return nom/denom
    
    def calc_distance_punish(self,k,l):
        """
        calculates the cost matrix of the particles to the template
        when punishing for particles in between

        Parameters
        ----------
        k : int
            template index
        l : int 
            particle index

        Returns
        -------
        distance : float
            cost of moving from l to k

        """
        distance=0
        coord1=self.particles[l,:]
        coord2=self.template[k,:]
        minx=np.min([coord1[0],coord2[0]])
        miny=np.min([coord1[1],coord2[1]])
        maxx=np.max([coord1[0],coord2[0]])
        maxy=np.max([coord1[1],coord2[1]])
        
        #if there are particles in the box around the start and end point
        # punish proportional to their distance from the line of transport
        for x in range(minx-2,maxx+2):
            for y in range(miny-2,maxy+2):
                if np.any((self.particles[:]==np.array([x,y])).all(1)):
                    dist_line=self.calc_distance_point_line(x,y,coord1[0],coord1[1],coord2[0],coord2[1])
                    dist_point1=np.sqrt((x-coord1[0])**2+(y-coord1[1])**2)
                    dist_point2=np.sqrt((x-coord2[0])**2+(y-coord2[1])**2)
                    if dist_line<min(dist_point1,dist_point2):
                        distance=distance+1/(dist_line+1)
        return distance
        
    def calc_costs(self,p,mode):
        """
        calculates the cost matrix of the particles to the template

        Parameters
        ----------
        p : float
            power parameter.
        mode : string
            choise of "lp" and "punish".

        Returns
        -------
        None.

        """
        if mode=="lp":
            for k in range(self.num_template):
                for l in range(self.num_particle):
                    self.cost_matrix[k,l]=self.calc_distance(p,k,l)
            #Ensure cost to dummy template points = 0
            for k in range(self.num_dummies):
                self.cost_matrix[k+self.num_template,:]=0
        if mode=="punish":
            lam=np.float64(0.005)
            for k in range(self.num_template):
                for l in range(self.num_particle):
                    self.cost_matrix[k,l]=self.calc_distance(p,k,l)
                    if not self.cost_matrix[k,l]==0 and self.cost_matrix[k,l]<6:
                        self.cost_matrix+=np.float64(lam*self.calc_distance_punish(k,l))
                    else:
                        #no long distances or self-assignments
                        self.cost_matrix+=5
            #Ensure cost to dummy template points = 0
            for k in range(self.num_dummies):
                self.cost_matrix[k+self.num_template,:]=0

    
    def switch_in_order_list(self,target_num_1,target_num_2):
        """
        switched the order of move executions in the order list
        between particle 1 and particle 2

        Parameters
        ----------
        target_num_1 : int
            target to be switched
        target_num_2 : int
            target to be switched

        Returns
        -------
        None.

        """
        pos_1= np.int64(np.where(self.order_of_execution==target_num_1))
        pos_2= np.int64(np.where(self.order_of_execution==target_num_2))
        if pos_2>pos_1:
            temp_pos=np.array(self.order_of_execution[pos_2])
            self.order_of_execution[pos_2]=np.array(self.order_of_execution[pos_1])
            self.order_of_execution[pos_1]=np.array(temp_pos)
    
    def create_order_list(self):
        """
        Rearanges the order list in such a way that if a target is already
        occupied by a particle and it is not the assigned particle, then the 
        move order between the target and the occupying particle's target switches
        

        Returns
        -------
        None.

        """
        for k in range(self.num_template):
            target_1_c=self.template[k,:]
            if any((self.particles[:]==target_1_c).all(1)):
                target_1=np.where((self.template[:]==target_1_c).all(1))[0]
                particle_2=np.where((self.particles[:]==target_1_c).all(1))
                if not (self.assignment[target_1,1]==particle_2).all():
                    target_2=np.where(self.assignment[:,1]==particle_2)[1]
                    self.switch_in_order_list(target_1, target_2)
    
    def calc_distance_coord(self,pos1,pos2,p):
        """
        calculates the l^p distance between pos1 and pos2

        Parameters
        ----------
        pos1 : np.array
            x and y positions of pos1
        pos2 : np.array 
            x and y positions of pos2
        p : float
            power for distance

        Returns
        -------
        float
            l^p distance between pos1 and pos2

        """
        return ((np.abs(pos1[0]-pos2[0]))**p+
                (np.abs(pos1[1]-pos2[1]))**p)**(1/p)
    
    
    def fill_from_center(self,p):
        """
        Returns the sorted indexes of the template points to fill from the center

        Parameters
        ----------
        p : float
            power for distance measure

        Returns
        -------
        None.

        """
        center=np.mean(self.template,0)
        dist_list=np.zeros([self.num_template,2])
        dist_list[:,0]=np.arange(0,self.num_template)
        for k in range(self.num_template):
            dist_list[k,1]=self.calc_distance_coord(center,self.template[k,:],p)
        sortedlist=dist_list[dist_list[:, 1].argsort()]
        self.order_of_execution=np.int64(sortedlist[:,0])
    
    def check_collision(self,start,target,route,particle_num):
        """
        checks collisions from start to target using route

        Parameters
        ----------
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]
        route : np array
            array describing the route
        particle_num : int
            number of start particle

        Returns
        -------
        collisions: np array
            array of particles colliding on the route
        num_collisions: int
            number of collisions on route

        """
        if np.all(start==target):
            if any((self.particles[:]==target).all(1)):
                print("Error: already a particle at target")
                return [],0
            else:
                return [],0
        else:
            #move right
            if route[0]==1:
                new_start=np.array(start)+np.array([1,0])
            #move down
            elif route[0]==2:
                new_start=np.array(start)+np.array([0,-1])
            #move left
            elif route[0]==3:
                new_start=np.array(start)+np.array([-1,0])
            #move up
            elif route[0]==4:
                new_start=np.array(start)+np.array([0,1])  
            collisions,num_collisions=self.check_collision(np.array(new_start),np.array(target),route[1:],particle_num)
        #if particle at start increase collisions
        if any((self.particles[:]==start).all(1)):
            collisions=np.int64(np.array(np.append(collisions,start)))
            num_collisions+=1
        return collisions,num_collisions
        
            
    def check_collisions(self,start,target):
        """
        Find the route from start to target with the least number of collisions

        Parameters
        ----------
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]

        Returns
        -------
        min_route : string
            route with minimal number of collisions, as locations
        min_route_array: np array
            route with minimal number of collisions, as directional array
        min_col: int
            number of collisions on minimal route

        """
        min_route=[]
        min_route_array=0
        min_col=1e99
        #find all possible direct routes
        route_array=self.possible_routes(start, target)
        particle_num=np.where((self.particles[:]==start).all(1))
        if (start==target).all():
            #particle already in place
            return min_route,[],0
        for k in range(len(route_array[:,0])):
            #check which particles collide and their number 
            collisions,num_collisions=self.check_collision(start,target,route_array[k,:],particle_num)
            #update if route is better
            if (num_collisions-1)<min_col:
                min_col=num_collisions-1
                min_route_array=route_array[k,:]
                min_route=np.reshape(collisions[:-2],[np.int64((len(collisions)-2)/2),2])
        #fill in the final route
        min_route=np.flip(np.vstack([target,min_route,start]),0)
        return min_route, min_route_array, min_col
    
    def check_available_spots(self,start,spots,mindim,maxdim,recdepth,maxdepth):
        """
        Checks which spots are reachable from start, calculated using recurrence

        Parameters
        ----------
        start : np array
            start position [x,y]
        spots : np array
            list of reachable spots so far
        mindim : int
            min x or min y to consider
        maxdim : int
            max x or max y to consider.
        recdepth : int
            current depth or recurrence of this function.
        maxdepth : int
            ax depth or recurrence of this function.

        Returns
        -------
        spots : np array
            list of reachable spots so far

        """
        
        if recdepth>maxdepth:
            return spots
        else:
            recdepth=recdepth+1
        new_start=start+np.array([1,0])
        if (not any((self.particles[:]==new_start).all(1))) and (not any((spots[:]==new_start).all(1))) and new_start[0]<maxdim and new_start[1]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        new_start=start+np.array([0,-1])
        if (not any((self.particles[:]==new_start).all(1))) and (not any((spots[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        new_start=start+np.array([-1,0])
        if (not any((self.particles[:]==new_start).all(1))) and (not any((spots[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        new_start=start+np.array([0,1])
        if (not any((self.particles[:]==new_start).all(1))) and (not any((spots[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        return spots
    
    def sort_move_lengths(self):
        """
        Sorts the moves in assignments by their lengths

        Returns
        -------
        target_move_length : np array
            

        """
        target_move_length=np.zeros([self.num_template,2])
        target_move_length[:,0]=self.assignment[:,0]
        target_move_length[:,1]=self.cost_matrix[self.assignment[:,0],self.assignment[:,1]]
        target_move_length=target_move_length[target_move_length[:,1].argsort()]
        return target_move_length
    
    def move_order(self,mode):
        """
        Creates a list of moves that needs to be executed according to some protocol

        Parameters
        ----------
        mode : string
            "shortest_first": first executes the shortest types of moves
            "fill_from_center": fills the center spots first
            "order_of_execution": executes the moves as given by the assignment
        Returns
        -------
        list
            starts: the start point of the moves 
            targets: the end points of the moves
            move_order_starts: the order of the start points from particles array
            move_order_Targets: the order of the end points from template
        
        """
        if mode=="shortest_first":
            target_move_length=self.sort_move_lengths()
            move_order_targets=np.int64(target_move_length[(target_move_length[:,1]>0),0])
            move_order_starts=self.assignment[move_order_targets,1]
            starts=self.particles[move_order_starts]
            targets=self.template[move_order_targets]
        elif mode=="fill_from_center":
            self.fill_from_center(1)
            self.create_order_list()
            move_order_targets=np.array(self.order_of_execution)
            move_order_starts=self.assignment[move_order_targets,1]
            starts=self.particles[move_order_starts]
            targets=self.template[move_order_targets]
        elif mode=="order_of_execution":
            self.create_order_list()
            move_order_targets=np.array(self.order_of_execution)
            move_order_starts=self.assignment[move_order_targets,1]
            starts=self.particles[move_order_starts]
            targets=self.template[move_order_targets]
        return [starts,targets,move_order_starts,move_order_targets]
    
    def coord_route_from_array(self,start,route):
        """
        from start and route as directional array creates route as locations

        Parameters
        ----------
        start : np array
            start position as [x,y]
        route : np array
            route described as array of the form [1,3,2,4,2,2,1]
            where 1=right, 2=down, 3=left, 4=up.

        Returns
        -------
        coord_route : np array
            array of locations describing the route

        """
        coord_route=np.zeros([len(route)+1,2])
        coord_route[0,:]=np.array(start)
        loc=np.array(start)
        for k in range(len(route)):
            if route[k]==1:
                loc+=np.array([1,0])
            elif route[k]==2:
                loc+=np.array([0,-1])
            elif route[k]==3:
                loc+=np.array([-1,0])
            elif route[k]==4:
                loc+=np.array([0,1])
            coord_route[k+1,:]=np.array(loc)
        return coord_route
        

    def execute_move(self,start,target,start_num,target_num,mode="no_coll"):
        """
        Moves an atom from start to target

        Parameters
        ----------
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]
        start_num : int
            atom number to move.
        target_num : int
            target spot to end
        mode : string, optional
            DESCRIPTION. The default is "no_coll".

        Returns
        -------
        None.

        """
        
        found=0
        if mode=="no_coll":
            if not np.all(start==target):
                pos_spots=self.check_available_spots_2(start, np.array([[-10,10],start]), -1, 20, 0, 5)
                if any((pos_spots[:]==target).all(1)):
                    min_route_array=self.route_shortener(self.available_spot_route(start, target, -2, 20, 0, 6)[0])
                    min_route=np.vstack([start,target])
                    min_col=0
                    found=1 # a route has been found
                    
        if found==0:
            min_route,min_route_array,min_col=self.check_collisions(start, target)
        dist_counter=0
        for k in range(len(min_route)-1):
            if found==0:
                routelength=int(np.abs(min_route[-(k+2),0]-min_route[-(k+1),0])+np.abs(min_route[-(k+2),1]-min_route[-(k+1),1]))
            else:
                routelength=len(min_route_array)
            self.AOM_path_new_pos=np.append(self.AOM_path_new_pos,np.append(np.zeros([1,routelength]),1))
            if dist_counter==0:
                self.AOM_path=np.vstack([self.AOM_path,self.coord_route_from_array(min_route[-(k+2),:],min_route_array[-routelength:])])
            else:
                self.AOM_path=np.vstack([self.AOM_path,self.coord_route_from_array(min_route[-(k+2),:],min_route_array[-(dist_counter+routelength):-dist_counter])])
            dist_counter+=routelength
        self.particles[start_num,:]=self.template[target_num,:]
        self.atoms[start_num].pos=self.template[target_num,:]*self.R
    
    def plot_grid_particles_template(self,ax):
        ax.scatter(self.grid[:,0],self.grid[:,1],c='k')
        ax.scatter(self.template[:,0],self.template[:,1],c='r')
        ax.scatter(self.particles[:,0],self.particles[:,1],c='b')
        ax.scatter(self.particles[np.where(np.int64(1-self.particle_alive)),0],self.particles[np.where(np.int64(1-self.particle_alive)),1],c='m')
        return ax

    def plot_reachable_states(self,ax):
        ax.scatter(self.grid[:,0],self.grid[:,1],c='g')
        return ax

    def plot_assignment(self,ax):
        """
        plots the found assignment in ax

        Parameters
        ----------
        ax : matplotlib Axes object
            ax on which to plot

        Returns
        -------
        ax : Axes object
            object plotted axes

        """
        for k in range(len(self.assignment)):
            ax.plot([self.template[self.assignment[k,0],0],self.particles[self.assignment[k,1],0]],
                    [self.template[self.assignment[k,0],1],self.particles[self.assignment[k,1],1]],c='c')
        return ax

    def plot_AOM(self,AOM_position,ax):  
        ax.scatter(AOM_position[0],AOM_position[1],c='g')
        return ax
    
    def plot_situation(self):
        fig,ax=plt.subplots()
        ax=self.plot_grid_particles_template(ax)
        ax=self.plot_assignment(ax)
        plt.show()
        
    def plot_AOM_movement(self):
        self.particles=np.array(self.particles_init)
        self.plot_situation()
        for k in range(len(self.AOM_path)):
            fig,ax=plt.subplots()
            AOM_position=np.array(self.AOM_path[k,:])
            particle=np.where((self.particles==AOM_position).all(axis=1))
            if np.random.rand()>self.succes_move_grid_1:
                self.particle_alive[particle]=0
            ax=self.plot_grid_particles_template(ax)
            #ax=self.plot_assignment(ax)
            ax=self.plot_AOM(AOM_position,ax)
            if not self.AOM_path_new_pos[k]:
                self.particles[particle,:]=np.int64(np.array(self.AOM_path[k+1,:]))
            else:
                if np.random.rand()>self.succes_move_total:
                    self.particle_alive[particle]=0
            plt.show()        
        
        
    def execute_reordering(self,order="shortest_first",move_coll="yes_coll",draw=0):
        """
        executes the sorting of the particles

        Parameters
        ----------
        order : string
            "shortest_first": first executes the shortest types of moves
            "fill_from_center": fills the center spots first
            "order_of_execution": executes the moves as given by the assignment
        move_coll : string, optional
            "no_coll" or "yes_coll". The default is "yes_coll".
            tries to avoid collisions by not moving direct if "no_coll"
        draw : boolean, optional
            draw the movement. The default is 0.

        Returns
        -------
        None.

        """
        starts,targets,start_nums,target_nums=self.move_order(order)
        for k in range(len(starts)):
            self.execute_move(starts[k],targets[k],start_nums[k],target_nums[k],move_coll)
            if draw:
                self.plot_situation()
        
        
    def check_available_spots_2(self,start,spots,mindim,maxdim,recdepth,maxdepth):
        """
        Checks which spots are reachable from start, calculated using recurrence

        Parameters
        ----------
        start : np array
            start position [x,y]
        spots : np array
            list of reachable spots so far
        mindim : int
            min x or min y to consider
        maxdim : int
            max x or max y to consider.
        recdepth : int
            current depth or recurrence of this function.
        maxdepth : int
            ax depth or recurrence of this function.

        Returns
        -------
        spots : np array
            list of reachable spots so far

        """
        
        if recdepth>maxdepth:
            return spots
        else:
            recdepth=recdepth+1
        #move right
        new_start=start+np.array([1,0])
        if (not any((self.particles[:]==new_start).all(1)))  and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots_2(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        #move down
        new_start=start+np.array([0,-1])
        if (not any((self.particles[:]==new_start).all(1)))  and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots_2(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        #move left
        new_start=start+np.array([-1,0])
        if (not any((self.particles[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots_2(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        #move up
        new_start=start+np.array([0,1])
        if (not any((self.particles[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            spots=np.vstack([spots,new_start])
            spots=self.check_available_spots_2(new_start,spots,mindim,maxdim,recdepth,maxdepth)
        return spots
    
    def find_available_spot_route(self,start,route,target,mindim,maxdim,recdepth,maxdepth):
        """
        tries to find a route from start to target by extending the supplied route

        Parameters
        ----------
        start : np array
            start position [x,y]
        route : int
            route described as integer of the form 1324221
            where 1=right, 2=down, 3=left, 4=up.
        target : np array
            end position [x,y]
        mindim : int
            min x or min y to consider
        maxdim : int
            max x or max y to consider.
        recdepth : int
            current depth or recurrence of this function.
        maxdepth : int
            ax depth or recurrence of this function.

        Returns
        -------
        route: int
            route described as integer of the form 1324221
            where 1=right, 2=down, 3=left, 4=up.
        done: int
            indicates whether the route finding has completed

        """
        #check if max recurrence depth has been reached
        if recdepth>maxdepth:
            return route,0
        else:
            recdepth=recdepth+1
        #check if route has been completed
        if np.all(start==target):
            return route,1
        #move right
        new_start=start+np.array([1,0])
        if (not any((self.particles[:]==new_start).all(1)))  and new_start[0]<maxdim and new_start[1]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            new_route=route*10+1
            final_route,done=self.find_available_spot_route(new_start,new_route,target,mindim,maxdim,recdepth,maxdepth)
            if done==1:
                return final_route,done
        #move down
        new_start=start+np.array([0,-1])
        if (not any((self.particles[:]==new_start).all(1)))  and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            new_route=route*10+2
            final_route,done=self.find_available_spot_route(new_start,new_route,target,mindim,maxdim,recdepth,maxdepth)
            if done==1:
                return final_route,done
        #move left
        new_start=start+np.array([-1,0])
        if (not any((self.particles[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            new_route=route*10+3         
            final_route,done=self.find_available_spot_route(new_start,new_route,target,mindim,maxdim,recdepth,maxdepth)
            if done==1:
                return final_route,done
        #move up
        new_start=start+np.array([0,1])
        if (not any((self.particles[:]==new_start).all(1))) and new_start[1]<maxdim and new_start[0]<maxdim and new_start[0]>mindim and new_start[1]>mindim:
            new_route=route*10+4        
            final_route,done=self.find_available_spot_route(new_start,new_route,target,mindim,maxdim,recdepth,maxdepth)
            if done==1:
                return final_route,done
        return route, 0
    
    def available_spot_route(self,start,target,mindim,maxdim,recdepth,maxdepth):
        """
        Finds a route from start to target

        Parameters
        ----------
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]
        mindim : int
            min x or min y to consider
        maxdim : int
            max x or max y to consider.
        recdepth : int
            current depth or recurrence of this function.
        maxdepth : int
            ax depth or recurrence of this function.

        Returns
        -------
        pos_routes : string
            route in coordinates where 1=right, 2=down, 3=left, 4=up.

        """
        pos_routes_int,done=self.find_available_spot_route(start, 0, target, mindim, maxdim, recdepth, maxdepth)
        #change from integer to string
        pos_routes=np.array([int(d) for d in ''.join(str(x) for x in [pos_routes_int])]).reshape(1,-1)
        return pos_routes
    
    def route_shortener(self,route):
        """
        shortens a route when possible

        Parameters
        ----------
        route : string
            route in coordinates where 1=right, 2=down, 3=left, 4=up.

        Returns
        -------
        string
            route in coordinates where 1=right, 2=down, 3=left, 4=up.

        """
        route_old=''.join(str(x) for x in route)
        done=0
        #shortens a route when possible
        while not done:
            route_new=route_old
            route_new=route_new.replace('13', '')
            route_new=route_new.replace('31', '')
            route_new=route_new.replace('24', '')
            route_new=route_new.replace('42', '')
            route_new=route_new.replace('123', '2')
            route_new=route_new.replace('143', '4')
            route_new=route_new.replace('321', '2')
            route_new=route_new.replace('341', '4')
            route_new=route_new.replace('234', '3')
            route_new=route_new.replace('214', '1')
            route_new=route_new.replace('432', '3') 
            route_new=route_new.replace('412', '1')
            if route_old==route_new:
                done=1
            else:
                route_old=route_new
        return np.array([int(d) for d in route_new])

    def possible_routes_as_ints(self,route_array,start,target):
        """
        recursively gives all the direct routes from start to target

        Parameters
        ----------
        route_array : int
            current route
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]

        Returns
        -------
        new routes
            routes 1 step further

        """
        #Go left and right
        if start[0]>target[0]:
            #Go left
            route_array_1=self.possible_routes_as_ints(route_array,start+np.array([-1,0]),target)
            route_array_1=np.array(route_array_1*10+3)
        elif start[0]<target[0]:
            #Go right
            route_array_1=self.possible_routes_as_ints(route_array,start+np.array([1,0]),target)
            route_array_1=np.array(route_array_1*10+1)
        else:
            route_array_1=[]
        #Go up and down
        if start[1]>target[1]:
            route_array_2=self.possible_routes_as_ints(route_array,start+np.array([0,-1]),target)
            route_array_2=np.array(route_array_2*10+2)
        elif start[1]<target[1]:
            route_array_2=self.possible_routes_as_ints(route_array,start+np.array([0,1]),target)
            route_array_2=np.array(route_array_2*10+4)
        else:
            route_array_2=[]
        if route_array_1==[] and route_array_2==[]:
            return route_array
        else:
            return np.int64(np.array(np.append(route_array_1,route_array_2)))
        
    def possible_routes(self,start,target):
        """
        finds the direct routes from start to target

        Parameters
        ----------
        start : np array
            start position [x,y]
        target : np array
            end position [x,y]

        Returns
        -------
        pos_routes : list
            list of possible routes from start to target

        """
        pos_routes_int=self.possible_routes_as_ints(0,start,target)
        if isinstance(pos_routes_int,int):
            numroutes=1
            #change from int to string
            pos_routes=np.array(int(d) for d in str(pos_routes_int))
        else:
            numroutes=len(pos_routes_int)
            #change from int to string
            pos_routes=np.array([int(d) for d in ''.join(str(x) for x in pos_routes_int)]).reshape(numroutes,-1)
        return pos_routes
        
    def create_assignment(self,mode):
        """
        Creates an assingment and inverse assignment on the route finder

        Parameters
        ----------
        mode : string
            one of "Browaeys_heuristic", "Hungarian" and "Sinkhorn".

        Returns
        -------
        None
        
        """
        if mode=="Browaeys_heuristic":
            self.create_assignment_browaeys_heuristic()
        elif mode=="Hungarian":
            self.create_assignment_hungarian()
        elif mode=="Sinkhorn":
            self.create_assignment_sinkhornn()
        self.inverse_assignment=self.assignment[self.assignment[:, 1].argsort()]

    def cost_of_assignment(self):
        """
        Calculates the cost of the assignment based on the cost matrix

        Returns
        -------
        sums : float
            Total cost of the assignment

        """
        sums=0
        for k in range(self.num_template):
            sums+=self.cost_matrix[self.assignment[k,0],self.assignment[k,1]]
        return sums
        
    def create_assignment_browaeys_heuristic(self):
        """
        Creates an assignment by leaving all spots which are already filled
        and then taking the minimal cost particle for every template spot

        Returns
        -------
        None.

        """
        assignment=np.zeros([2,0])
        takentraps=[]
        takenparticles=[]
        #Leave all particles which are already in template spots
        zero_assignment=np.array(np.where(self.cost_matrix[range(self.num_template),:]==0))
        takentraps=np.append(takentraps,zero_assignment[0,:])
        cost_matrix_2=np.array(self.cost_matrix)
        cost_matrix_2[:,zero_assignment[1,:]]=1e99
        assignment=np.append(assignment,np.transpose(zero_assignment))
        #For all template spots not filled, look at the minimal cost particle to fill
        for k in np.setdiff1d(range(self.num_template),takentraps):
            min_particle=np.argmin(cost_matrix_2[k,:])
            cost_matrix_2[:,min_particle]=1e99
            assignment=np.append(assignment,[[k],[min_particle]])
        assignment=np.reshape(assignment,[self.num_template,2])    
        self.assignment=np.int64(assignment[assignment[:,0].argsort()])
        
    def create_assignment_hungarian(self):
        """
        Creates an assignment based on the Hungarian/linear sum assignment algorithm
        from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

        Returns
        -------
        None.

        """
        targets,particles=sci.linear_sum_assignment(self.cost_matrix)
        self.assignment=np.transpose(np.array([targets[0:self.num_template],particles[0:self.num_template]]))
   
    def create_assignment_sinkhornn(self):
        """
        Create an assignment based on the sinkhorn algorithm from
        https://pythonot.github.io/auto_examples/plot_OT_1D.html

        Returns
        -------
        None.

        """
        #Equal weights for all particles and template spots
        weights1=np.zeros([self.num_template])+1/self.num_template
        weights2=np.zeros([self.num_particle])+1/self.num_particle

        #Calculating the Sinkhornn solution
        reg=0.0001 #Regularization parameter
        gamma=ot.sinkhorn(weights1, weights2, self.cost_matrix[0:self.num_template,0:self.num_particle], reg)
        
        #Creating the assignment by picking the maximal values
        assignment=np.zeros([2,0])
        for k in range(self.num_template):
            min_particle=np.argmax(gamma[k,:])
            gamma[:,min_particle]=-1e99 #ensure that template spot can't be picked again
            assignment=np.append(assignment,[[k],[min_particle]])
        assignment=np.reshape(assignment,[self.num_template,2])    
        self.assignment=np.int64(assignment[assignment[:,0].argsort()])
        

        
        
    
        
    
    
    
    
            
            
        
            
        