import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time
import matplotlib.pyplot as plt
 

class TriangLattice:

    def __init__(self, Npoints,  normed):

        self.Npoints = Npoints
        self.a =np.array([[1,0],[1/2,np.sqrt(3)/2]])  #original graphene lattice vectors: rows are basis vectors
        self.b =(2*np.pi)*np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]]) # original graphene reciprocal lattice vectors : rows are basis vectors
        self.normed=normed
        self.GM=np.sqrt(self.b[0,:].T@self.b[0,:])
        self.GMvec=[self.b[0,:],self.b[1,:]]
        self.LMvec=[self.a[0,:],self.a[1,:]]
        #some symmetries:
        #C2z
        th1=np.pi
        self.C2z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C3z
        th1=2*np.pi/3
        self.C3z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C2x inv
        self.C2x=np.array([[1,0],[0,-1]]) #rotation matrix 
        self.VolMBZ=self.Vol_MBZ()


   

    def __repr__(self):
        return "lattice( LX={w}, twist_angle={c})".format(h=self.Npoints, c=self.normed)

    #hexagon where the pointy side is up
    def hexagon1(self,pos,Radius_inscribed_hex):
        x,y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters #effective rotation
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge
    
    #hexagon where the flat side is up
    def hexagon2(self,pos,Radius_inscribed_hex):
        y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

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
        K=Vertices_list[0::2]
        Kp=Vertices_list[1::2]
        M=Edges_list[0::2]
        Mp=Edges_list[1::2]

        return Vertices_list, Gamma, K, Kp, M, Mp


    
    #same as Generate lattice but for the original graphene (FBZ of triangular lattice)
    def Generate_lattice(self):

        
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizex


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        for x in nn1:
            for y in nn2:
                kx=(2*np.pi*x/LP)
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))
                if self.hexagon1( ( kx, ky ), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)

        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        nn_1pp=np.array(nn_1p)
        nn_2pp=np.array(nn_2p)

        KX=(2*np.pi*nn_1pp/LP)
        KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))

        # #Making the sampling lattice commensurate with the MBZ
        # fact=K[1][0]/np.max(KX)
        # KX=KX*fact
        # KY=KY*fact
        
        return [KX,KY]

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
        VV, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)
        VV=np.array(VV+[VV[0]]) #verices

        L=[]
        # L=L+[K[0]]+[Gamma]+[M[0]]+[Kp[-1]] ##path in reciprocal space
        L=L+[K[0]]+[Gamma]+[M[0]]+[K[0]] ##path in reciprocal space Andrei paper

        Nt_points=40
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
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return np.array(Vertices_list+[Vertices_list[0]])/Gnorm

    ### SYMMETRY OPERATIONS ON THE LATTICE
    def C2zLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc2z=KX*self.C2z[0,0]+KY*self.C2z[0,1]
        KYc2z=KX*self.C2z[1,0]+KY*self.C2z[1,1]
        Indc2z=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc2z[i]=np.argmin( (KX-KXc2z[i])**2 +(KY-KYc2z[i])**2)

        return [KXc2z,KYc2z, Indc2z]

    def C2xLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc2x=KX*self.C2x[0,0]+KY*self.C2x[0,1]
        KYc2x=KX*self.C2x[1,0]+KY*self.C2x[1,1]
        Indc2x=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc2x[i]=np.argmin( (KX-KXc2x[i])**2 +(KY-KYc2x[i])**2)

        return [KXc2x,KYc2x, Indc2x]

    def C3zLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc3z=KX*self.C3z[0,0]+KY*self.C3z[0,1]
        KYc3z=KX*self.C3z[1,0]+KY*self.C3z[1,1]
        Indc3z=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc3z[i]=np.argmin( (KX-KXc3z[i])**2 +(KY-KYc3z[i])**2)

        return [KXc3z,KYc3z, Indc3z]

    def mask_KPs(self, KX,KY, thres):
        [GM1,GM2]=self.GMvec
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        Radius_inscribed_hex=1.0000005*k_window_sizey
        K=np.sqrt(KX**2+KY**2)
        ind=np.where(K<k_window_sizex*thres)
        return [KX[ind],KY[ind], ind]
    
    def Umklapp_List(self, umklapps):
        #G processes
        G=self.GM
        Gu=[]
        [GM1, GM2]=self.GMvec
        for i in range(-10,10):
            for j in range(-10,10):
                Gp=i*GM1+j*GM2
                Gpn=np.sqrt(Gp.T@Gp)
                if  Gpn<=G*(umklapps+0.1):
                    Gu=Gu+[[i,j]]
        #             plt.scatter(Gp[0], Gp[1], c='r')
        #         else:
        #             plt.scatter(Gp[0], Gp[1], c='b')

        # thetas=np.linspace(0,2*np.pi, 100)
        # xx=umklapps*G*np.cos(thetas)
        # yy=umklapps*G*np.sin(thetas)
        # plt.plot(xx,yy)
        # plt.savefig("ulat.png")
        # plt.close()
        return Gu
    
    def Generate_Umklapp_lattice(self, KX, KY, numklaps):
        Gu=self.Umklapp_List(numklaps)
        [GM1, GM2]=self.GMvec
        KXu=[]
        KYu=[]
        
        for GG in Gu:
            KXu=KXu+[KX+GG[0]*GM1[0]+GG[1]*GM2[0]]
            KYu=KYu+[KY+GG[0]*GM1[1]+GG[1]*GM2[1]]
        
        KXum=np.concatenate( KXu )
        KYum=np.concatenate( KYu )
        return [KXum, KYum]
    
    def insertion_index(self, KX,KY, KQX,KQY):
        #list of size Npoi that has the index of K in KQ
        Npoi=np.size(KX)
        Ik=[]
        for j in range(Npoi):
            indmin=np.argmin(np.sqrt((KQX-KX[j])**2+(KQY-KY[j])**2))
            Ik.append(indmin)
        return Ik

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