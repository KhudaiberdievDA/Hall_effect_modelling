import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import scipy.sparse.linalg  as sp_lalg
import random
import time
import scipy.optimize as sp_opt


def Geometry():
    points_from_file = np.loadtxt('Mesa_structure_geometry.txt',dtype = [('f0',float),('f1',float)])
    Edge_points = [(round(points_from_file[i][0]-points_from_file[0][0]),round(points_from_file[i][1]-points_from_file[0][1])) for i in range(1,len(points_from_file))]

    Nx_grid = 756
    Ny_grid = 148
    hx,hy = 1,1#mkm

    Type_of_a_point = np.zeros((Ny_grid, Nx_grid),dtype=int) #! 0 - outsise, 1 - Neumann, 2 - Dirichlet, 3 - inside
    Point_to_line_Neumann = np.empty((Ny_grid, Nx_grid),dtype=int)
    Point_to_line_Dirichlet = np.empty((Ny_grid, Nx_grid))

    class Edge_line:
        x1, y1 = 0, 0 #x y are the same as i j 
        x2, y2 = 0, 0
        orientation = 0 # 0 - horisontal, 1 - vertical
        Boundary_cond = 1 # 1 - Neymann, 2 - Dirikhle
        y_i, x_i = [], []
        num_of_Dirichlet = 0
        num_of_all_lines = 0
        Num_Fi1 = []
        Num_Fi2 = []
        resistance = 0

        def __init__(self,r1,r2, Boundary_cond = 1) -> None:
            self.x1, self.y1 = r1[0], r1[1]
            self.x2, self.y2 = r2[0], r2[1]
            if self.y1 == self.y2 and self.x1 != self.x2:
                self.orientation = 0 #vertical
            elif self.x1 == self.x2 and self.y1 != self.y2:
                self.orientation = 1 #horizontal

            self.Boundary_cond = Boundary_cond
            self.x_i,self.y_i = [], []
            #let's mark points on the field 
            for i in range(len(self)+1):
                self.y_i.append(self.y1 + (self.y2-self.y1)//(len(self))*i)
                self.x_i.append(self.x1 + (self.x2-self.x1)//(len(self))*i)
                if  Type_of_a_point[self.y_i[i], self.x_i[i]] != 2:
                    Type_of_a_point[self.y_i[i], self.x_i[i]] = Boundary_cond
            if Boundary_cond == 2:
                for i in range(len(self)+1):
                    Point_to_line_Dirichlet[self.y_i[i], self.x_i[i]] = Edge_line.num_of_Dirichlet
                Edge_line.num_of_Dirichlet += 1
            if Boundary_cond == 1:
                for i in range(len(self)+1):
                    Point_to_line_Neumann[self.y_i[i], self.x_i[i]] = Edge_line.num_of_all_lines
            Edge_line.num_of_all_lines += 1
            self.Num_Fi1 = np.zeros(len(self) + 1,dtype=int)
            self.Num_Fi2 = np.zeros(len(self) + 1,dtype=int)

            self.resistance = 0

        def __len__(self):
            return np.abs(self.x1 - self.x2) + np.abs(self.y1 - self.y2) #LOL, THATS HOW I DEFINE LENGTH

        def On_the_segment(self, x0, y0):
            if_crosses_line = y0*(self.x2-self.x1) == (self.y2 - self.y1)*x0 + self.y1*self.x1 - self.y2*self.x1
            if_on_segment_x = np.abs(self.x1-x0) + np.abs(self.x2 - x0) == np.abs(self.x1 - self.x2)
            if_on_segment_y = np.abs(self.y1-y0) + np.abs(self.y2 - y0) == np.abs(self.y1 - self.y2) 
            return if_crosses_line and if_on_segment_y and if_on_segment_x

        def Crosses_negative_line(self,x0,y0): #we want to determine if this line crosses x-x0<0
            if self.On_the_segment(x0,y0):
                print("This little buddy is on the Edge, not in the Volume")
                return np.NaN
            elif self.orientation == 1 and (self.x1-x0)<0: 
                if (self.y1-y0)*(self.y2-y0)<0:
                    return 1
                elif self.y1-y0 == 0 or self.y2-y0 == 0:
                    return np.heaviside(self.y1-y0 + self.y2-y0,0)
            return 0
 
    Type_of_a_line = np.ones(len(Edge_points)-1)
    Which_are_Dirichlet = [14,18,42,46,61,76,80,104,108,123]
    for i in Which_are_Dirichlet:
        Type_of_a_line[i] = 2
    Full_Edge = [Edge_line(Edge_points[i],Edge_points[i+1],Boundary_cond = Type_of_a_line[i]) for i in range(len(Edge_points)-1)]

    def Is_point_in_mesa(x,y):
        return sum([line.Crosses_negative_line(x,y) for line in Full_Edge]) % 2

    # for i in range(Nx_grid):
    #     for j in range(Ny_grid):
    #         if Type_of_a_point[j,i] == 0:
    #             Type_of_a_point[j,i] = 3*Is_point_in_mesa(i,j)
    # with open('Geomerty_Type_of_a_point.txt', 'wb') as f:
    #     np.save(f, Type_of_a_point)
    with open('Geomerty_Type_of_a_point.txt', 'rb') as f:
        Type_of_a_point = np.load(f) 

    def plot_geometry():
        plt.imshow(Type_of_a_point)
        plt.show()
    plot_geometry()
    
    return Edge_line, Full_Edge, Type_of_a_line, Type_of_a_point, Nx_grid, Ny_grid, Point_to_line_Dirichlet, Which_are_Dirichlet, Point_to_line_Neumann



Edge_line, Full_Edge, Type_of_a_line, Type_of_a_point, Nx_grid, Ny_grid, Point_to_line_Dirichlet, Which_are_Dirichlet, Point_to_line_Neumann = Geometry() 

