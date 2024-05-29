# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:12:26 2023

@author: s165827
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:28:29 2023

@author: s165827
"""
import unittest
import matplotlib.pyplot as plt
import astar
import scipy.optimize as sci
import numpy as np
import random

class BasicTests(unittest.TestCase):

    def test_bestpath(self):
        
        def create_square_template(dim_in,low):
            """
            Creates a square template of dim_in x dim_in spots with the left bottom
            corner at [low,low]

            Parameters
            ----------
            dim_in : int
                dimension of the grid
            low : int
                left corner located at [low,low]

            Returns
            -------
            template: np.array of ints with dimension dim_in**2 x 2 
                array with all the locations of the template

            """
            template=np.zeros([dim_in**2,2])
            for k in range(dim_in):
                for l in range(dim_in):
                    template[dim_in*k+l,:]=[int(low+k),int(low+l)]
            template=np.int64(template)
            return template

        def create_square_grid(dim_out):
            """
            creates a grid of possible spots as dim_out x dim_out

            Parameters
            ----------
            dim_out : int
                dimension of the square grid

            Returns
            -------
            grid: np.array of ints with dimension dim_out**2 x 2 
                array with all the locations of the grid

            """
            grid=np.zeros([dim_out**2,2])
            for k in range(dim_out):
                for l in range(dim_out):
                    grid[dim_out*k+l,:]=[k,l]
            grid=np.int64(grid)
            return grid
        
        def create_start(template,grid):
            template=string_to_matrix(template)
            num_traps=len(grid)
            num_atoms=len(template)
            initially_filled_traps=np.random.permutation(range(num_traps))[range(num_atoms)]
            print(grid[initially_filled_traps,:])
            return grid[initially_filled_traps,:]
        
        def calc_costs(start,template):
            start=string_to_matrix(start)
            template=string_to_matrix(template)
            cost_matrix=np.zeros([len(start),len(template)])
            for k in range(len(start)):
                for l in range(len(template)):
                    cost_matrix[k,l]=np.linalg.norm(start[k,:]-template[l,:])**2
            return cost_matrix
        
        def create_assignment_hungarian(cost_matrix):
            targets,particles=sci.linear_sum_assignment(cost_matrix)
            assignment=np.transpose(np.array([targets[0:len(cost_matrix)],particles[0:len(cost_matrix)]]))
            return assignment
            
        def matrix_to_string(a):
            return a.tobytes()
        
        def string_to_matrix(string):
            mat=np.frombuffer(string,int)
            if len(mat)==32:
                return np.frombuffer(string,int).reshape([16,2])
            else:
                return np.frombuffer(string,int).reshape([16,4])[:,[0,2]]
        
        def ar_plot(start,template,grid,assignment):
            plt.title(cost(matrix_to_string(start),template))
            template=string_to_matrix(template)
            plt.scatter(grid[:,0],grid[:,1],color='black')
            plt.scatter(template[:,0],template[:,1],color='red')
            plt.scatter(start[:,0],start[:,1],color='blue')
            for k in range(len(start)):
                plt.arrow(start[k,0],start[k,1],template[assignment[k,1],0]-start[k,0],template[assignment[k,1],1]-start[k,1])
            plt.show()
        
        def already_filled(pos,start):
            if any((start[:]==pos).all(1)):
                return 1
            else:
                return 0
        
        def neighbors(start):
            neighborlist=[]
            start=string_to_matrix(start)
            for k in range(start.shape[0]):
                new_array=np.array(start)
                if start[k,0]+1<dim_out:
                    if not already_filled(start[k,:]+[1,0],start):
                        new_array[k,:]=start[k,:]+[1,0]
                        neighborlist+=[matrix_to_string(np.array(new_array))]
                ar_plot(np.array(new_array),template,grid,assignment)
                new_array=np.array(start)
                if start[k,0]-1>=0:
                    if not already_filled(start[k,:]+[-1,0],start):
                        new_array[k,:]=start[k,:]+[-1,0]
                        neighborlist+=[matrix_to_string(np.array(new_array))]
                new_array=np.array(start)
                if start[k,1]+1<dim_out:
                    if not already_filled(start[k,:]+[0,1],start):
                        new_array[k,:]=start[k,:]+[0,1]
                        neighborlist+=[matrix_to_string(np.array(new_array))]
                new_array=np.array(start)
                if start[k,1]-1>=0:
                    if not already_filled(start[k,:]+[0,-1],start):
                        new_array[k,:]=start[k,:]+[0,-1]
                        neighborlist+=[matrix_to_string(np.array(new_array))]
            np.random.shuffle(neighborlist)
            return neighborlist
        
        def goal_reached(start,template):
            if np.all(start==template):
                return True
            else:
                return False
        
        def distance(start,neighbour):
            return 1
        
        def cost(start,template):
            start=string_to_matrix(start)
            template=string_to_matrix(template)
            cost=0
            for k in range(len(start)):
                cost+=np.abs(start[k,0]-template[assignment[k,1],0])+np.abs(start[k,1]-template[assignment[k,1],1])
            return cost
        
        dim_out=8
        grid=create_square_grid(dim_out)

        template=matrix_to_string(create_square_template(4, 1))
        print(string_to_matrix(template))
        start=matrix_to_string(create_start(template, grid))
        print(string_to_matrix(start))
        print("------")

        cost_matrix=calc_costs(start,template)
        assignment=create_assignment_hungarian(cost_matrix)
        print(assignment)

        path = list(astar.find_path(start, template, neighbors_fnct=neighbors, reversePath=True,
                    heuristic_cost_estimate_fnct=cost, distance_between_fnct=distance, is_goal_reached_fnct=goal_reached))
        print(path)

if __name__ == '__main__':
    unittest.main()