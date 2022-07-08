import chunk
import numpy as np
import Lattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import Lattice
from scipy.interpolate import interp1d
from scipy.linalg import circulant
import sys  
from time import sleep
from parfor import parfor
from scipy.optimize import minimize

class Ham():
    def __init__(self, hbvf, latt, rescale=None):
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale
        
        self.hvkd = self.rescale*hbvf
        self.latt=latt
        
        
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION
    
    def eigens(self, kx,ky, Mz=0):
        [GM1,GM2]=self.latt.LMvec
        
        Mzev=Mz*self.hvkd 

        e1=(GM1+GM2)/3
        e2=(-2*GM1+GM2)/3
        e3=(-2*GM2+GM1)/3
        
        
        W3=self.hvkd #0.00375/3  #in ev
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3))
        hk=W3*ee
        hk_n=np.sqrt( np.abs(hk)**2 +Mzev**2 )
        
        #wavefunctions
        invsqrt=np.sqrt( 1/2 - Mzev/(2*hk_n) )
        subsM=Mzev-hk_n
        sqrt2=np.sqrt(2)
        vup_1=invsqrt*subsM/(hk*sqrt2)
        vup_2=invsqrt
        vdown_1=np.conj(hk)/(sqrt2*(Mzev*subsM+np.abs(hk)**2 ))
        vdown_2=invsqrt
        
        # print(k@e1,k@e2,k@e3,hk_n, 2*np.pi*1/3)
        psi1=np.array([vdown_1, vdown_2])
        psi2=np.array([vup_1, vup_2])
        
        
        psi=np.zeros([np.size(psi1), 2])+0*1j
        psi[:,0]=psi1.flatten()
        psi[:,1]=psi2.flatten()

        
        return np.array([-hk_n,+hk_n ]), psi
    
    def eigens2(self, kx,ky, Mz=0):
        [GM1,GM2]=self.latt.LMvec
        
        Mzev=Mz*self.hvkd 

        e1=(GM1+GM2)/3
        e2=(-2*GM1+GM2)/3
        e3=(-2*GM2+GM1)/3
        
        
        W3=self.hvkd #0.00375/3  #in ev
        k=np.array([kx,ky])
        
        ee=(np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3))
        hk=W3*ee
        hk_n=np.sqrt( np.abs(hk)**2 +Mzev**2 )
        
        return np.array([-hk_n,+hk_n ])
    
    def ExtendE(self,E_k , umklapp):
        Gu=self.latt.Umklapp_List(umklapp)
        
        Elist=[]
        for GG in Gu:
            Elist=Elist+[E_k]
            
        return np.vstack(Elist)


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
        
        self.maxfil=4
    
    
    def precompute_E_psi_1v(self,Mz=0):
    
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz):
            E1,wave1=self.hpl.eigens(self.KX1bz[l],self.KY1bz[l],Mz)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])

        
        return [psi_plus,Ene_valley_plus]
    
    def precompute_E_psi_1v_v2(self,Mz=0):
        
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        @parfor(range(self.Npoi1bz), (0,), bar=False)
        def fun(l, mu):
            E1=self.hpl.eigens2(self.KX1bz[l],self.KY1bz[l],Mz)
            return E1
        Ene_valley_plus_a=np.array(fun)
        
        # for l in range(self.Npoi1bz):
        #     E1=self.hpl.eigens2(self.KX1bz[l],self.KY1bz[l],Mz)
        #     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)

        
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])

        
        return Ene_valley_plus


    ###########DOS FOR DEBUGGING

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

    ### FERMI SURFACE ANALYSIS


    def High_symmetry(self,Mz=0):
        Ene_valley_plus_a=np.empty((0))
        psi_plus_a=[]

        nbands=self.nbands
        kpath=self.latt.High_symmetry_path()

        Npoi=np.shape(kpath)[0]
        for l in range(Npoi):
            # h.umklapp_lattice()
            # break
            E1p,wave1p=self.hpl.eigens(kpath[l,0],kpath[l,1],Mz=0)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
            psi_plus_a.append(wave1p)


        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
    
        print("the shape of the energy array is",np.shape(Ene_valley_plus))
        qa=np.linspace(0,1,Npoi)
        for i in range(nbands):
            plt.plot(qa,Ene_valley_plus[:,i] , c='b')
        plt.xlim([0,1])
        # plt.ylim([-0.009,0.009])
        plt.savefig("highsym.png")
        plt.close()
        return [Ene_valley_plus]
    
    
    #energy calculation
    
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
        
    
    def calc_energy_MZ(self,MZ,T, mu):
        T_ev=T*self.hpl.hvkd 
        mu_ev=mu*self.hpl.hvkd 
        U=3*self.hpl.hvkd  #to reproduce liang-fu;s calculation
        # [ppp,Ene_valley_plus_dos]=self.precompute_E_psi_1v(MZ)
        Ene_valley_plus_dos=self.precompute_E_psi_1v_v2(MZ)
        [earr, dos_arr, f2 ]=self.DOS(Ene_valley_plus_dos)
        de=earr[1]-earr[0]
        inte=np.trapz(dos_arr*earr*self.nf(earr-mu_ev,T_ev))*de +(MZ*self.hpl.hvkd )**2 /U
        print(MZ, inte)
        plt.plot(earr, dos_arr)
        plt.scatter(earr, dos_arr, s=1)
        plt.savefig("dos1.png")
        plt.close()
        return inte
    
    
    
    def calc_filling_args_MZ(self,mu, MZ,T, earr, dos_arr):
        T_ev=T*self.hpl.hvkd 
        mu_ev=mu*self.hpl.hvkd 
        de=earr[1]-earr[0]
        inte=np.trapz(dos_arr*self.nf(earr-mu_ev,T_ev))*de
        # print(MZ, inte)
   
        return inte-self.maxfil/2
    
    
    def chem_for_filling_lin(self, fill, earr, dos_arr,MZ,T):
        
        
        # n=[]
        # for e in earr:
        #     ff=self.calc_filling_args_MZ(e, MZ,T, earr, dos_arr)
        #     n.append(ff)
        
        @parfor(earr, (0,), bar=False)
        def fun(e, mu):
            ff=self.calc_filling_args_MZ(e, MZ,T, earr, dos_arr)
            return ff
        n=np.array(fun)
        
        # mine=earr[1]
        # maxe=earr[-2]
        # mu=self.bisection(self.calc_filling_args_MZ,mine,maxe,50, MZ, T, earr, dos_arr)
        
        mu_ind=np.argmin((np.array(n)-fill)**2)
        mu=earr[mu_ind]
        nfil=self.calc_filling_args_MZ(mu, MZ,T, earr, dos_arr)
        if fill==0.0:
            mu=0.0
            nfil=0.0
         
        if fill>0:
            errfil=abs((nfil-fill)/fill)
            print(errfil, "error in filling")
            if errfil>0.1:
                print("TOO MUCH ERROR IN THE FILLING CALCULATION") 
            
        return [mu, nfil]
    
    def calc_energy_MZ_fixed_filling(self,MZ,T, fil):
        T_ev=T*self.hpl.hvkd 
        U=3*self.hpl.hvkd  #to reproduce liang-fu;s calculation
        Ene_valley_plus_dos=self.precompute_E_psi_1v_v2(MZ)
        [earr, dos_arr, f2 ]=self.DOS(Ene_valley_plus_dos)
        
        
        [mu_ev, nfil]=self.chem_for_filling_lin( fil, earr, dos_arr,MZ, T_ev)
        
            
        print("the filling is actually",nfil,fil, mu_ev)
        de=earr[1]-earr[0]
        inte=np.trapz(dos_arr*earr*self.nf(earr-mu_ev,T_ev))*de +(MZ*self.hpl.hvkd )**2 /U
        print(MZ, inte)
        plt.plot(earr, dos_arr)
        plt.scatter(earr, dos_arr, s=1)
        plt.savefig("dos1.png")
        plt.close()
        return inte
    


def main() -> int:
    ##when we use this main, we are exclusively testing the moire hamiltonian symmetries and methods
    from scipy import linalg as la
    

    
    try:
        filling_index=int(sys.argv[1]) 

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")


    try:
        Nsamp=int(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")


    ##########################################
    #parameters energy calculation
    ##########################################

    a_graphene=2.46*(1e-10) #in meters
    nbands=2 
    Nsamp=int(sys.argv[2])
    #Lattice generation
    latt=Lattice.TriangLattice(Nsamp,0)
    
    #parameters for the bandstructure
    hbvf = 1#2.1354; # eV
    BW=6
    print("hbvf is ..",hbvf )
    
    #generating the dispersion 
    resc=1 #0.001
    h=Ham(hbvf, latt, resc)
    disp=Dispersion( latt, nbands, h,h)
    disp.High_symmetry()
    
    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    disp=Dispersion( latt, nbands, h, h)
    Nfils=5
    [fillings,mu_values]=disp.mu_filling_array(Nfils, False, False, True) #read write calculate
    filling_index=int(sys.argv[1]) 
    mu=mu_values[filling_index]
    filling=fillings[filling_index]
    print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    print(mu_values,fillings)
    
    MZ=0.0
    T=.15
    mu=0
    fills=[0, 0.005,0.01,0.02, 0.025, 0.03]
    s=time.time()
    Energy_calc=disp.calc_energy_MZ(MZ,T, mu, )
    e=time.time()
    print(Energy_calc, "time for energy calc", e-s)
    
    
    #the lines of code below minimize the mean field hamiltonian
    M_list=[]
    MZ=0.
    TT=np.linspace(0.01,0.25,10)*BW
    # TT=[0.001]
    
    for T in TT:
        s=time.time()
        res=minimize(disp.calc_energy_MZ, MZ, args=(T,mu), method='COBYLA')
        e=time.time()
        MZ=res.x
        M_list.append(MZ)
        print(T,MZ,disp.calc_energy_MZ( MZ,T, mu ), "time for minimization", e-s)
    
    M=np.array(M_list)/BW
    T=TT/BW
    plt.plot(T,M)
    plt.scatter(T,M)
    ml=np.max(M)
    plt.ylim([0-0.02*ml,ml+0.02*ml])
    
    plt.savefig("MzT2.png")
    plt.close()
    
    
    
    # #the lines of code below minimize the mean field hamiltonian
    # M_list=[]
    # MZ=0.
    # TT=np.linspace(0.01,0.25,10)*BW
    # fil=1
    
    # for T in TT:
    #     s=time.time()
    #     res=minimize(disp.calc_energy_MZ_fixed_filling, MZ, args=(T,fil), method='COBYLA', options={'maxiter':20})
    #     e=time.time()
    #     MZ=res.x
    #     M_list.append(MZ)
    #     print(T,MZ,disp.calc_energy_MZ_fixed_filling(MZ,T, fil), "time for minimization", e-s)
    
    # M=np.array(M_list)/BW
    # T=TT/BW
    # plt.plot(T,M)
    # plt.scatter(T,M)
    # ml=np.max(M)
    # plt.ylim([0-0.02*ml,ml+0.02*ml])
    
    # plt.savefig("MzT2fixed.png")
    # plt.close()

            
if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
