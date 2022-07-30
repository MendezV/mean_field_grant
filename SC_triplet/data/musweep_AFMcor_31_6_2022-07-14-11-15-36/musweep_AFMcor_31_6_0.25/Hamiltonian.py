import numpy as np
import scipy
import matplotlib.pyplot as plt
import Lattice
from parfor import parfor
from scipy.optimize import minimize
import time
from scipy.interpolate import interp1d
import pandas as pd

class Ham():
    def __init__(self, t, latt, mu, rescale=None):
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale
        
        self.t = self.rescale*t
        self.latt=latt
        self.mu=mu
            
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)

    #######################################################
    ########### METHODS FOR CALCULATING THE DISPERSION
    #######################################################
    
    def eigens(self, kx,ky, Delt=0):
        [GM1,GM2]=self.latt.LMvec
        
        Delt_ev=Delt*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)-self.mu
        
        
        return [hk, hk]
    
    def eigens2(self, kx,ky, Delt=0):
        [GM1,GM2]=self.latt.LMvec
        
        Delt_ev=Delt*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)
        
        
        return hk
    
    def eigens_ND(self, kx,ky, Delt=0):
        [GM1,GM2]=self.latt.LMvec
        

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)
        
        
        return [-hk,-hk,hk,hk]
 
    def eigens_nsp(self, kx,ky, Delt=0):
        [GM1,GM2]=self.latt.LMvec
        
        Delt_ev=Delt*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)-self.mu
        
        dis1=np.sqrt(hk**2 + 2*Delt**2 )
        
        return [-dis1,-dis1,dis1,dis1]
        
    def eigens_sp(self, kx,ky, Delt=0):
        [GM1,GM2]=self.latt.LMvec
        
        Delt_ev=Delt*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)-self.mu
        
        dis1=np.sqrt(hk**2 + 4*Delt**2 )
        dis2=np.sqrt( hk**2 )
        
        return [-dis1,-dis2,dis2,dis1]
    
    def eigens_psp(self, kx,ky, Delt=np.array([0,0,0])):
        [GM1,GM2]=self.latt.LMvec
        
        Delt_ev=Delt*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        hk=-self.t*np.real(ee)-self.mu
        
        
        dis1=np.sqrt(hk**2 + 2*Delt[0]**2 *(1+2*np.cos(Delt[1])*np.sin(Delt[1])*np.sin(Delt[2])) )
        dis2=np.sqrt(hk**2 + 2*Delt[0]**2 *(1-2*np.cos(Delt[1])*np.sin(Delt[1])*np.sin(Delt[2])) )
        
        return [-dis1,-dis2,dis2,dis1]
      
    def eigens_AFM(self, kx,ky, MZ=0):
        [GM1,GM2]=self.latt.LMvec
        
        MZ_ev=MZ*self.t

        e1=GM1
        e2=GM2
        
        k=np.array([kx,ky])
        Q=np.array([np.pi,np.pi])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2))
        eeQ=(np.exp(1j*(k+Q)@e1)+np.exp(1j*(k+Q)@e2))
        hk=-self.t*np.real(ee)-self.mu
        hkQ=-self.t*np.real(eeQ)-self.mu
        
        
        dis1=np.sqrt( 0.25*((hk-hkQ)**2) +MZ_ev**2 )+0.5*(hk+hkQ)
        dis2=-np.sqrt( 0.25*((hk-hkQ)**2) +MZ_ev**2 )+0.5*(hk+hkQ)
        
        return [dis2,dis1,dis1,dis2]
 

class Dispersion():
    
    def __init__(self, latt, nbands, hpl, hmin):

        self.lat=latt
        
        #changes to eliominate arg KX KY put KX1bz first, then call umklapp lattice and put slef in KX, KY
        #change to implement dos 
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        self.Npoi1bz=np.size(self.KX1bz)
        self.latt=latt
        
        [self.KX1bz_h, self.KY1bz_h]=latt.Generate_lattice_half()
        self.Npoi1bz_h=np.size(self.KX1bz_h)
        
        [self.KX1bz_h2, self.KY1bz_h2]=latt.Generate_lattice_half_2()
        self.Npoi1bz_h2=np.size(self.KX1bz_h2)
        
        self.maxfil=2
    
    def precompute_E_psi_1v(self,Delt=0):
    
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz):
            E1,wave1=self.hpl.eigens(self.KX1bz[l],self.KY1bz[l],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])

        
        return [psi_plus,Ene_valley_plus]
    
    def precompute_E_psi_1v_v2(self,Delt=0):
        
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        # @parfor(range(self.Npoi1bz), (0,), bar=False)
        # def fun(l, mu):
        #     E1=self.hpl.eigens2(self.KX1bz[l],self.KY1bz[l],Delt)
        #     return E1
        # Ene_valley_plus_a=np.array(fun)
        
        for l in range(self.Npoi1bz):
            E1=self.hpl.eigens2(self.KX1bz[l],self.KY1bz[l],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)

        
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])

        
        return Ene_valley_plus
     
    def precompute_E_psi_2v(self,Delt, var_epsilon_pl,var_epsilon_min):
        
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz):
            E1=var_epsilon_pl(self.KX1bz[l],self.KY1bz[l],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            
            E1=var_epsilon_min(self.KX1bz[l],self.KY1bz[l],Delt)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)

        
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi1bz,self.nbands])

        
        return [Ene_valley_plus,Ene_valley_min]
    
    def precompute_E_psi_half_2v(self,Delt, var_epsilon_pl,var_epsilon_min):
        
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz_h):
            E1=var_epsilon_pl(self.KX1bz_h[l],self.KY1bz_h[l],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            
            E1=var_epsilon_min(self.KX1bz_h[l],self.KY1bz_h[l],Delt)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)

        
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz_h,self.nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi1bz_h,self.nbands])

        
        return [Ene_valley_plus,Ene_valley_min]
    
    def precompute_E_psi_half_2_2v(self,Delt, var_epsilon_pl,var_epsilon_min):
        
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz_h2):
            E1=var_epsilon_pl(self.KX1bz_h2[l],self.KY1bz_h2[l],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            
            E1=var_epsilon_min(self.KX1bz_h2[l],self.KY1bz_h2[l],Delt)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)

        
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz_h2,self.nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi1bz_h2,self.nbands])

        
        return [Ene_valley_plus,Ene_valley_min]
    

    #################################
    ###########DOS and filling
    ################################# 

    def DOS(self,Ene_valley_plus_pre):
        Ene_valley_plus=Ene_valley_plus_pre
        nbands=np.shape(Ene_valley_plus)[1]
        print("number of bands in density of states calculation," ,nbands)
        eps_l=[]
        for i in range(nbands):
            eps_l.append(np.mean( np.abs( np.diff( Ene_valley_plus[:,i].flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)*5
        print("and epsilon is ...", eps)
        
        mmin=np.min(Ene_valley_plus)
        mmax=np.max(Ene_valley_plus)
        
        print("THE BANDWIDTH IS....", mmax-mmin)
        
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        binn=np.linspace(mmin,mmax,NN+1)
        valt=np.zeros(NN)

        val_p,bins_p=np.histogram(Ene_valley_plus[:,0].flatten(), bins=binn,density=True)
        valt=valt+val_p
        val_p,bins_p=np.histogram(Ene_valley_plus[:,1].flatten(), bins=binn,density=True)
        valt=valt+val_p
        
        bins=(binn[:-1]+binn[1:])/2
        
        
        valt=2*valt
        f2 = interp1d(binn[:-1],valt, kind='cubic')
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        
        return [bins,valt,f2 ]
    
    def bisection(self,f,a,b,N, *args):
        '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

        Parameters
        ----------
        f : function
            The function for which we are trying to approximate a solution f(x)=0.
        a,b : numbers
            The interval in which to search for a solution. The function returns
            None if f(a)*f(b) >= 0 since a solution is not guaranteed.
        N : (positive) integer
            The number of iterations to implement.

        Returns
        -------
        x_N : number
            The midpoint of the Nth interval computed by the bisection method. The
            initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
            midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
            If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
            iteration, the bisection method fails and return None.

        Examples
        --------
        >>> f = lambda x: x**2 - x - 1
        >>> bisection(f,1,2,25)
        1.618033990263939
        >>> f = lambda x: (2*x - 1)*(x - 3)
        >>> bisection(f,0,1,10)
        0.5
        '''
        if f(a,*args)*f(b,*args) >= 0:
            print("Bisection method fails.",f(a,*args),f(b,*args))
            return None
        a_n = a
        b_n = b
        for n in range(1,N+1):
            m_n = (a_n + b_n)/2
            f_m_n = f(m_n,*args)
            if f(a_n,*args)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            elif f(b_n,*args)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            elif f_m_n == 0:
                print("Found exact solution.")
                return m_n
            else:
                print("Bisection method fails.")
                return None
        return (a_n + b_n)/2

    def chem_for_filling(self, fill, f2, earr):
        
        NN=10000
        mine=earr[1]
        maxe=earr[-2]
        mus=np.linspace(mine,maxe, NN)
        dosarr=f2(mus)
        de=mus[1]-mus[0]
        
        
        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(NN):
            N=np.trapz(dosarr[0:mu_ind])*de
            ndens.append(N)
            
        nn=np.array(ndens)
        nn=self.maxfil*(nn/nn[-1])  - self.maxfil/2



        
        fn = interp1d(mus,nn-fill, kind='cubic')
        fn2 = interp1d(mus,nn, kind='cubic')
        
        mu=self.bisection(fn,mine,maxe,50)
        nfil=fn2(mu)
        if fill==0.0:
            mu=0.0
            nfil=0.0
         
        if fill>0:
            errfil=abs((nfil-fill)/fill)
            if errfil>0.1:
                print("TOO MUCH ERROR IN THE FILLING CALCULATION") 
            
        return [mu, nfil, mus,nn]
       
    def mu_filling_array(self, Nfil, read, write, calculate):
        
        fillings=np.linspace(0,self.maxfil/2-0.05*self.maxfil,Nfil)
        
        if calculate:
            [psi_plus,Ene_valley_plus_dos]=self.precompute_E_psi_1v()
        if read:
            print("Loading  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
                
        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos)

        mu_values=[]        
        for fill in fillings:
            [mu, nfil, es,nn]=self.chem_for_filling( fill, f2, earr)
            mu_values.append(mu)

        return [fillings,np.array(mu_values)]
    
    def dos_filling_array(self, Nfil, read, write, calculate):
        
        fillings=np.linspace(0,3.9,Nfil)
        
        if calculate:
            [psi_plus,Ene_valley_plus_dos]=self.precompute_E_psi_1v()
        if read:
            print("Loading  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)

    
        if write:
            print("saving  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)

        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos)


        
        return [earr, dos]

    #################################
    ########### FERMI SURFACE ANALYSIS
    #################################
    
    def High_symmetry(self,Delt, fdisp, id=''):
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]

        nbands=self.nbands
        kpath=self.latt.High_symmetry_path()

        Npoi=np.shape(kpath)[0]
        for l in range(Npoi):

            E1p= fdisp(kpath[l,0],kpath[l,1],Delt)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)


        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
    
        print("the shape of the energy array for High symmetry cut is",np.shape(Ene_valley_plus))
        qa=np.linspace(0,1,Npoi)
        for i in range(nbands):
            plt.plot(qa,Ene_valley_plus[:,i] , c='b')
        plt.xlim([0,1])
        # plt.ylim([-0.009,0.009])
        plt.savefig("highsym_"+id+".png")
        plt.close()
        return [Ene_valley_plus]
    
    def plot_FS(self,Delt, fdisp,mu, id):
        
        sq_kpoints=500
        xlist = np.linspace(-np.pi, np.pi, sq_kpoints)
        ylist = np.linspace(-np.pi, np.pi, sq_kpoints)
        X, Y = np.meshgrid(xlist, ylist)
        
        fig,ax=plt.subplots(1,1)
        
        XX=fdisp(X,Y)
        
        Z1 =  XX[0]
        Z2 =  XX[1]

        cp = ax.contour(X, Y, Z1,levels=[mu],colors='r')
        cp = ax.contour(X, Y, Z2,levels=[mu],colors='b')
        
        plt.savefig("FS_"+id+".png")
        plt.close()
        return None
    #################################
    ########### ENERGY CALCULATION
    #################################
    
    def nf(self, e, T):
        
        """[summary]
        fermi occupation function with a truncation if the argument exceeds 
        double precission

        Args:
            e ([double]): [description] energy
            T ([double]): [description] temperature

        Returns:
            [double]: [description] value of the fermi function 
        """
        Tp=T+1e-17 #To capture zero temperature
        rat=np.abs(np.max(e/Tp))
        
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)
    
    def calc_free_energy_metal(self,Delt,T, J):
        T_ev=T*self.hpl.t 
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(Delt,self.hpl.eigens,self.hmin.eigens)
        Elam1=Ene_valley_plus
        Elam2=Ene_valley_min
        F1=-T_ev*np.sum(np.log(1+np.exp(-Elam1/T_ev)))
        F2=-T_ev*np.sum(np.log(1+np.exp(-Elam2/T_ev)))
        F=F1+F2
        print(Delt, F)
        return F
    
    def calc_zero_energy_metal(self):
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(0,self.hpl.eigens,self.hmin.eigens)
        F=np.sum(Ene_valley_plus+Ene_valley_min)
        return F
      
    def calc_free_energy_nsp(self,Delt,T, J):
        T_ev=T*self.hpl.t 
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(Delt,self.hpl.eigens_nsp,self.hmin.eigens_nsp)
        Elam1=(Ene_valley_plus[:,int(self.nbands/2):])*0.5
        Elam2=(Ene_valley_min[:,int(self.nbands/2):])*0.5
        F1=-T_ev*np.sum(np.log(1+np.exp(-Elam1/T_ev)))
        F2=-T_ev*np.sum(np.log(1+np.exp(-Elam2/T_ev)))
        N=np.size(Ene_valley_plus)
        F=F1+F2+4*N*(Delt*self.hmin.t )**2 /J
        print(Delt, F)
        return F

    def calc_free_energy_sp(self,Delt,T, J):
        T_ev=T*self.hpl.t 
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(Delt,self.hpl.eigens_sp,self.hmin.eigens_sp)
        Elam1=(Ene_valley_plus[:,int(self.nbands/2):])*0.5
        Elam2=(Ene_valley_min[:,int(self.nbands/2):])*0.5
        F1=-T_ev*np.sum(np.log(1+np.exp(-Elam1/T_ev)))
        F2=-T_ev*np.sum(np.log(1+np.exp(-Elam2/T_ev)))
        N=np.size(Ene_valley_plus)
        F=F1+F2+4*N*(Delt*self.hmin.t )**2 /J
        print(Delt, F)
        return F
    
    def calc_free_energy_psp(self,Delt,T, J):
        T_ev=T*self.hpl.t 
        # [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(Delt,self.hpl.eigens_psp,self.hmin.eigens_psp)
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_half_2_2v(Delt,self.hpl.eigens_psp,self.hmin.eigens_psp)
        Elam1=Ene_valley_plus
        Elam2=Ene_valley_min
        F1=-T_ev*np.sum(np.log(1+np.exp(-Elam1/T_ev)))
        F2=-T_ev*np.sum(np.log(1+np.exp(-Elam2/T_ev)))
        N=2*np.size(Ene_valley_plus) #full BZ
        F=F1+F2+4*N*(Delt[0]*self.hmin.t )**2 /J
        print(Delt, F)
        return F
    
    def calc_free_energy_AFM(self,MZ,T, J):
        T_ev=T*self.hpl.t 
        # [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_2v(MZ,self.hpl.eigens_AFM,self.hmin.eigens_AFM)
        [Ene_valley_plus,Ene_valley_min]=self.precompute_E_psi_half_2v(MZ,self.hpl.eigens_AFM,self.hmin.eigens_AFM)
        Elam1=Ene_valley_plus
        Elam2=Ene_valley_min
        F1=-T_ev*np.sum(np.log(1+np.exp(-Elam1/T_ev)))
        F2=-T_ev*np.sum(np.log(1+np.exp(-Elam2/T_ev)))
        N=2*np.size(Ene_valley_plus) #full BZ
        F=F1+F2+N*(MZ*self.hmin.t )**2 /J
        print(MZ, F)
        return F



def main() -> int:
    
    #####
    # Parameters Diag: samples
    ####
    try:
        Npoints=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")

    #####
    # Electron parameters: mu J
    ####
    
    try:
        mu=float(sys.argv[2]) 

    except (ValueError, IndexError):
        raise Exception("Input float in the second argument to choose chemical potential for desired filling")
    
    try:
        J=float(sys.argv[3]) 


    except (ValueError, IndexError):
        raise Exception("Input float in the third argument to choose the interaction strenght J")


    latt=Lattice.SquareLattice( Npoints,  0)
    [KX,KY]=latt.Generate_lattice()
    print(np.size(KX), Npoints**2)
    
    ###hamiltonian
    t=1
    nbands=4
    hp=Ham(t,latt,mu)
    disp=Dispersion(latt,nbands, hp, hp)
    
    #test parameters and testing energy calculations
    T=.15
    
    dispM=Dispersion(latt,2, hp, hp)
    Delt=[0.1,0,0]
    dispM.High_symmetry(Delt, hp.eigens, 'metal')
    Delt=[0.0,0,0]
    Energy_calc=dispM.calc_free_energy_metal(Delt,T, J)
    print('free energy metal \n',Energy_calc)
    
    Delt=[0.1,0,0]
    disp.High_symmetry(Delt, hp.eigens_psp, 'SC')
    Delt=[0.0,0,0]
    Energy_calc_sc_0=disp.calc_free_energy_psp(Delt,T, J)
    print('free energy psp \n',Energy_calc_sc_0)
    
    MZ=0.1
    disp.High_symmetry(MZ, hp.eigens_AFM,'afm')
    MZ=0.0
    Energy_calc_AFM_0=disp.calc_free_energy_AFM(MZ,T, J)
    print('free energy AFM \n',Energy_calc_AFM_0)


    #the lines of code below minimize the mean field hamiltonian
    Delt_list=[]
    phi_list=[]
    theta_list=[]
    MZ_list=[]
    
    FEne_psp0_list=[]
    FEne_psp_list=[]
    
    FEne_AFM0_list=[]
    FEne_AFM_list=[]
    # TT=np.linspace(0.005,.04,50)[::-1]
    TT=np.linspace(0.01,0.5,50)[::-1]
    # TT=[0.001]
    
    #seed
    Delt=[0.2,0,0]
    MZ=0.2
    
    for T in TT:
        
        #zerp values at this T
        Energy_calc_sc_0=disp.calc_free_energy_psp([0,0,0],T, J)
        Energy_calc_AFM_0=disp.calc_free_energy_AFM(0,T, J)
        
        FEne_psp0_list.append(Energy_calc_sc_0)
        FEne_AFM0_list.append(Energy_calc_AFM_0)
        
        #minimization SC
        s=time.time()
        res=minimize(disp.calc_free_energy_psp, Delt, args=(T, J), method='COBYLA')
        Delt=res.x
        Fres=res.fun
        e=time.time()
        
        
        Delt_list.append(np.abs(Delt[0]))
        phi_list.append(np.abs(Delt[1]))
        theta_list.append(np.abs(Delt[2]))
        FEne_psp_list.append(Fres)
        
        #minimization AFM
        s=time.time()
        res=minimize(disp.calc_free_energy_AFM, MZ, args=(T, J), method='COBYLA')
        MZ=res.x
        Fres=res.fun
        e=time.time()
        
        
        MZ_list.append(np.abs(MZ))
        FEne_AFM_list.append(Fres)

        print(T,Delt,Fres, "time for minimization", e-s)
    
    
    
    #saving data
    Delts=np.array(Delt_list)
    phis=np.array(phi_list)
    thets=np.array(theta_list)
    MZs=np.array(MZ_list)
    
    FEne_psp0=np.array(FEne_psp0_list)
    FEne_psp=np.array(FEne_psp_list)
    
    FEne_AFM0=np.array(FEne_AFM0_list)
    FEne_AFM=np.array(FEne_AFM_list)
    
    one=np.ones(np.size(MZs))
    
    # print(np.size(TT),np.size(mu*one),np.size(J*one),np.size(Delts),np.size(phis),np.size(thets),np.size(MZs),np.size(FEne_psp0),np.size(FEne_psp))
    
    df = pd.DataFrame({'T':TT, 'mu':mu*one, 'J':J*one,'D':Delts, 'phi':phis,'theta':thets,'MZ':MZs,'FSC0':FEne_psp0,'FSC':FEne_psp,'FAFM0':FEne_AFM0,'FAFM':FEne_AFM, 'L':Npoints*one})
    df.to_hdf('data_mu_'+str(mu)+'_J_'+str(J)+'.h5', key='df', mode='w')
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit



    
    