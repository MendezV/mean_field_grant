import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time
import matplotlib.pyplot as plt

class SquareLattice:

    def __init__(self, Npoints,  normed):

        self.Npoints = Npoints
        self.a =np.array([[1,0],[0,1]])  #original graphene lattice vectors: rows are basis vectors
        self.b =(2*np.pi)*np.array([[1,0],[0,1]]) # original graphene reciprocal lattice vectors : rows are basis vectors
        self.normed=normed
        self.GM=np.sqrt(self.b[0,:].T@self.b[0,:])
        self.GMvec=[self.b[0,:],self.b[1,:]]
        self.LMvec=[self.a[0,:],self.a[1,:]]
        #some symmetries:
        #C2z
        th1=np.pi
        self.C2z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C4z
        th1=2*np.pi/4
        self.C4z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        
        #C8z
        th1=2*np.pi/8
        self.C8z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C2x inv
        self.C2x=np.array([[1,0],[0,-1]]) #rotation matrix 
        self.VolMBZ=self.Vol_MBZ()
        
    def __repr__(self):
        return "lattice( LX={w}, twist_angle={c})".format(h=self.Npoints, c=self.normed)

    #gets high symmetry points
    
    def FBZ_points(self,b_1,b_2):
        #creating reciprocal lattice
        Np=4
        n1=np.arange(-Np,Np+1)
        n2=np.arange(-Np,Np+1)
        Recip_lat=[]
        for i in n1:
            for j in n2:
                point=b_1*i+b_2*j
                Recip_lat.append(point)

        #getting the nearest neighbours to the gamma point
        Recip_lat_arr=np.array(Recip_lat)
        dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
        sorted_dist=np.sort(list(set(dist)) )
        points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

        #getting the voronoi decomposition of the gamma point and the nearest neighbours
        vor = Voronoi(points)
        Vertices=(vor.vertices)

        #ordering the points counterclockwise in the -pi,pi range
        angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
        Vertices_list=list(Vertices)

        # joint sorting the two lists for angles and vertices for convenience later.
        # the linear plot routine requires the points to be in order
        # atan2 takes into acount quadrant to get the sign of the angle
        angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

        ##getting the M points as the average of consecutive K- Kp points
        Edges_list=[]
        for i in range(len(Vertices_list)):
            Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

        Gamma=[0,0]
        M=Vertices_list
        X=Edges_list

        return Vertices_list, Gamma, M,X

    #Generate BZ k points
    
    def Generate_lattice(self):

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP//2,LP//2,1)*2*np.pi/LP
        nn2=np.arange(-LP//2,LP//2,1)*2*np.pi/LP

        KX,KY=np.meshgrid(nn1,nn2)
        
        return [KX.flatten(),KY.flatten()]
    
    def Generate_lattice_half(self):

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP//2,LP//2,1)*np.sqrt(2)*np.pi/LP
        nn2=np.arange(-LP//2,LP//2,1)*np.sqrt(2)*np.pi/LP

        KXp,KYp=np.meshgrid(nn1,nn2)
        KX=self.C8z[0,0]*KXp+self.C8z[0,1]*KYp
        KY=self.C8z[1,0]*KXp+self.C8z[1,1]*KYp
        
        return [KX.flatten()[::2],KY.flatten()[::2]]
    
    #normal linear interpolation to generate samples accross High symmetry points
     
    def linpam(self,Kps,Npoints_q):
        Npoints=len(Kps)
        t=np.linspace(0, 1, Npoints_q)
        linparam=np.zeros([Npoints_q*(Npoints-1),2])
        for i in range(Npoints-1):
            linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
            linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

        return linparam
        
    def High_symmetry_path(self):
        [GM1,GM2]=self.GMvec
        VV, Gamma, M,X=self.FBZ_points(GM1,GM2)
        VV=np.array(VV+[VV[0]]) #verices

        L=[]
        L=L+[Gamma]+[M[0]]+[X[0]]+[Gamma] ##path in reciprocal space Andrei paper

        Nt_points=200
        kp_path=self.linpam(L,Nt_points)
        
        # plt.scatter(kp_path[:,0],kp_path[:,1])
        # plt.plot(VV[:,0], VV[:,1])
        # plt.savefig("highSpath.png")
        # plt.close()
        

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return kp_path/Gnorm
    
    def boundary(self):
        [GM1,GM2]=self.GMvec
        Vertices_list, Gamma, M,X=self.FBZ_points(GM1,GM2)

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return np.array(Vertices_list+[Vertices_list[0]])/Gnorm
    
    #FBZ volume
    
    def Vol_MBZ(self):
        [GM1,GM2]=self.GMvec
        zhat=np.array([0,0,1])
        b_1=np.array([GM1[0],GM1[1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([GM2[0],GM2[1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec
    
    #WZ volume
    
    def Vol_WZ(self):
        [LM1,LM2]=self.LMvec
        zhat=np.array([0,0,1])
        b_1=np.array([LM1[0],LM1[1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([LM2[0],LM2[1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec
    
def main() -> int:
    Npoints=100
    sq=SquareLattice( Npoints,  0)
    [KX,KY]=sq.Generate_lattice()
    plt.scatter(KX,KY)
    plt.show()
    k=sq.High_symmetry_path()
    
    [KX,KY]=sq.Generate_lattice_half()
    plt.scatter(KX,KY)
    plt.show()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
