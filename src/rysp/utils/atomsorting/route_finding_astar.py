# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:28:29 2023

@author: s165827
"""
import unittest
import matplotlib.pyplot as plt
import astar
import numpy as np

class BasicTests(unittest.TestCase):

    def test_bestpath(self):
        """ensure that we take the shortest path, and not the path with less elements.
           the path with less elements is A -> B with a distance of 100
           the shortest path is A -> C -> D -> B with a distance of 60
        """
    
        
        def create_template(dim_out, dim_in):
            grid=np.zeros([dim_out,dim_out],int)
            start=int(np.floor((dim_out-dim_in)/2))
            stop=int(np.floor((dim_out+dim_in)/2))
            grid[start:stop,start:stop]=1
            return grid
    
        def create_start(template):
            template=string_to_matrix(template)
            total=int(np.sum(template))
            spaces=int(template.shape[0]*template.shape[1])
            filled=np.random.permutation(range(spaces))[range(total)]
            start=np.zeros(spaces,int)
            start[filled]=1
            start=np.reshape(start,template.shape)
            return start
        
        def matrix_to_string(a):
            return a.tobytes()
        
        def string_to_matrix(string):
            return np.frombuffer(string,int).reshape([6,6])
        
        def ar_plot(mat):
            plt.matshow(mat)
            plt.title(cost2(mat,template))
            plt.show()
        
        def neighbors(start):
            neighborlist=[]
            start=string_to_matrix(start)
            for k in range(start.shape[0]):
                for l in range(start.shape[1]):
                    if start[k,l]==1:
                        new_array=np.array(start)
                        if k-1>=0:
                            if start[k-1,l]==0:
                                new_array[k,l]=0
                                new_array[k-1,l]=1
                                neighborlist+=[matrix_to_string(np.array(new_array))]
                        ar_plot(np.array(new_array))
                        new_array=np.array(start)
                        if k+1<start.shape[0]:
                            if start[k+1,l]==0:
                                new_array[k,l]=0
                                new_array[k+1,l]=1
                                neighborlist+=[matrix_to_string(np.array(new_array))]
                        new_array=np.array(start)
                        if l-1<start.shape[1]:
                            if start[k,l-1]==0:
                                new_array[k,l]=0
                                new_array[k,l-1]=1
                                neighborlist+=[matrix_to_string(np.array(new_array))]
                        new_array=np.array(start)
                        if l+1<start.shape[1]:
                            if start[k,l+1]==0:
                                new_array[k,l]=0
                                new_array[k,l+1]=1
                                neighborlist+=[matrix_to_string(np.array(new_array))]
            return neighborlist
        
        def goal_reached(start,template):
            if np.all(start==template):
                return True
            else:
                return False
        
        def distance(start,neighbour):
            return 1
        
        def cost(start, template):
            template=string_to_matrix(template)
            start=string_to_matrix(start)
            cost_mat=start-template
            cost_mat[cost_mat<0]=0
            cost=np.sum(cost_mat)
            return cost
        
        def cost3(start,template):
            pos=np.where(start)
            return np.linalg.norm(pos[0]-4)**2+np.linalg.norm(pos[1]-4)**2
        
        def cost2(start,template):
            template=string_to_matrix(template)
            start=string_to_matrix(start)
            cost_mat=start-template
            cost_mat[cost_mat<0]=0
            points_template=np.where(template)
            points_cost=np.where(cost_mat)
            cost=0
            for k in range(len(points_cost[0])):
                cost+=np.amin(np.abs(points_cost[0][k]-points_template[0])+np.abs(points_cost[1][k]-points_template[1]))
            cost=cost+0.1*cost3(start,template)
            print(cost)
            return cost
        
        template=matrix_to_string(create_template(6,3))
        start=matrix_to_string(create_start(template))
        shape=[4,4]
        path = list(astar.find_path(start, template, neighbors_fnct=neighbors,
                    heuristic_cost_estimate_fnct=cost, distance_between_fnct=distance, is_goal_reached_fnct=goal_reached))
        print(path)

if __name__ == '__main__':
    unittest.main()