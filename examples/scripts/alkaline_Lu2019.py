import matplotlib.pyplot as plt
import numpy as np
from scikits.umfpack import spsolve
from scipy.sparse import coo_matrix

N=2 #initial problem rectilinear
Ncath = 5 #standard cathode problem
NJ=100
itmax=500
tol=1e-6
# ====== Discharge info
I = 0.12285*5 # A (amps) battery discharge current
# ====== General cathode info
ep = 0.274 # initial porosity of cathode
sigma = 20.0 # S/cm graphite conductivity
H = 6.0*2.54 # cm battery inside height
W = 3.0*2.54 # cm electrode plate width
t2 = 0.78 # transferrence number wrt vol av velocity of OHV_
V_alpha = 20.45 # cm3/mol molar vol MnOOH
V_gamma = 17.29 # cm3/mol molar vol MnO2
MWmn = 86.94 # g/mol MW of MnO2
dmn = MWmn/V_gamma # g/cm3 density of MnO2
V_0 = 18.07 # cm3/mol molar vol H2O
V_a = 17.8 # cm3/mol molar vol KOH
initKOH = 0.009 # mol/cm3 initial KOH conc

ro_Teflon = 1 # g/cm3 density of colloidal Teflon
ro_MnO2 = 5.026 # g/cm3 density of manganese dioxide
ro_KS44 = 2.26 # g/cm3 density of ks44
x_Teflon = 0.05 # mass fraction (Teflon)
x_MnO2 = 0.65 # mass fraction(MnO2)
x_KS44 = 0.30 # mass fraction(ks44)
ro_mix = ((x_Teflon/ro_Teflon)+(x_MnO2/ro_MnO2)+(x_KS44/ro_KS44))**(-1) #density mixture
L_c = 0.7
V_c = L_c*H*W
Vc = V_c*(1-ep)
m_cathode = Vc*ro_mix
# ====== General anode info
epa = 0.266 # initial porosity of anode (Chen 0.75)
V_Zn = 9.15 # cm3/mol molar vol Zn
V_ZnO = 14.51 # cm3/mol molar vol ZnO
theta = 0.8 # mixed reaction parameter
ro_Zn = 7.14 # g/cm3 density of zinc
ro_ZnO = 5.61 # g/cm3 density of zinc oxide
x_Zn = 0.85 # mass fraction Zn
x_ZnO = 0.10 # mass fraction ZnO
roa_mix = ((x_Teflon/ro_Teflon)+(x_Zn/ro_Zn)+(x_ZnO/ro_ZnO))**(-1) #density of mixtured anode
m_anode = 430 # g mass of anode
Va = m_anode/roa_mix # cm3 anode volume
V_a = Va/(1-epa) # thickness of anode
L_a = V_a/H/W

# ====== MnO2 particle info
r_ri = 0.005 # cm init MnO2 av particle radius
W1 = 0.65*m_cathode # g total mass of MnO2 in cathode
MW = 86.9368 # g/mol MW of MnO2
DH = 6e-10 # cm2/s proton diffusion coeff in solid
Vpart_i = (4.0/3)*3.14*r_ri**3 # cm3 init volume of MnO2 particle
Vc = H*W*L_c
NN = W1/Vc/Vpart_i/dmn # cm-3 number of MnO2 particles per cath vol
concMn = W/MW/Vc # mol/cm3 total conc of MnO2 in cathode
romax = (r_ri**3*V_alpha/V_gamma)**(1./3.) # cm max particle radius
Asep = H*W
# ====== Kinetic info
io = 0.0000002 # A/cm2 exch current density
ac = 0.5 # cathodic symmetry parameter
F = 96500.0 # C/mol faraday's number
R = 8.314 # J/mol-K gas constant
T = 298.0 # K temp
ba = (1.0-ac)*F/R/T # 1/V anodic kinetic constant
bc = ac*F/R/T # 1/V cathodic kinetic constantm2 separator area
h=L_c/float(NJ-1)
# region for the problem
rr = np.linspace(0,L_c,NJ)
# ===========
# INITIAL GUESSES
# ===========
def initguess():
    cold = np.ones([N,NJ])
    cold[0,:]=-0.05
    cold[1,:]=0.1
    return cold
# INTERPOLATE
# ===========
def interpolate( c, h):
    ( N, NJ )=c.shape
    cE=np.concatenate( ( c[:,1:NJ], np.zeros((N,1)) ), axis=1 )
    cW=np.concatenate( ( np.zeros((N,1)), c[:,0:NJ-1] ), axis=1 )
    dcdx=(cE-cW)/2.0/h
    dcdx[:, 0 ] = ( - 3.0*c[:, 0 ] + 4.0*c[:,1 ] - c[:,2 ] ) /2.0 /h
    dcdx[:,NJ-1] = ( 3.0*c[:,NJ-1] - 4.0*c[:,NJ-2] + c[:,NJ-3] ) /2.0 /h
    d2cdx2= (cE + cW - 2*c)/h**2
    d2cdx2[:,0] = np.zeros((1,N))
    d2cdx2[:,NJ-1] = np.zeros((1,N))
    return dcdx , d2cdx2
# FILLMAT
# ===========
def fillmat1(cold,dcdx,d2cdx2):
    #________first column refers to equation
    #________second column refers to position
    #________third column refers to species
    ( N, NJ )=cold.shape
    sma=np.zeros((N,NJ,N))
    smb=np.zeros((N,NJ,N))
    smd=np.zeros((N,NJ,N))
    smg=np.zeros((N,NJ))
    # Calculate varying physical params
    kappa = 0.45 # S/cm electrolyte conductivity
    # N particles per volume will not change. However particle radius will
    # This will affect area per volume a
    r_r = r_ri
    Apart = 4.0*3.14*r_r**2 # cm2 area of MnO2 particle
    a = NN*Apart # cm-1 area per volume MnO2 in cathode
    aio = a*io # A/cm3 exchange transfer current
    # lim will initially be very large because ro and ri are equal
    # We begin by setting it to a high number
    ep = 0.4
    # Cathode Model is entered HERE

    # Initial, steady-state problem:
    smb[0,:,0] = 1.0
    smd[0,:,1] = -1.0/kappa/ep**1.5 - 1.0/sigma
    smg[0,:] = -dcdx[0,:] + (1.0/kappa/ep**1.5 + 1.0/sigma)*cold[1,:] - \
    I/H/W/sigma
    smd[1,:,0] = aio*(-ba*np.exp(ba*cold[0,:]) - bc*np.exp(-bc*cold[0,:]))
    smb[1,:,1] = 1.0
    smg[1,:] = -dcdx[1,:] + \
    aio*np.exp(ba*cold[0,:]) - aio*np.exp(-bc*cold[0,:])
    #__________________________________________________ Boundary-Condition 1
    smp = np.zeros([N,N])
    sme = np.zeros([N,N])
    smf = np.zeros([N,1])
    sme[1,1] = 1.0
    smf[1] = I/H/W - cold[1,0]
    # Non-B.C.
    smp[0,0] = 1.0
    sme[0,1] = -1.0/kappa/ep**1.5 - 1.0/sigma
    smf[0] = -dcdx[0,0] + (1.0/kappa/ep**1.5 + 1.0/sigma)*cold[1,0] - \
    I/H/W/sigma

    # Insert (sme smp smf) into (smb smd smg)
    smb[:,0,:] = smp[:,:]
    smd[:,0,:] = sme[:,:]
    smg[:,0] = np.transpose(smf)
    #_______________________________________________ Boundary-Condition 2
    sme = np.zeros([N,N])
    smp = np.zeros([N,N])
    smf = np.zeros([N,1])
    # B.C.
    sme[1,1] = 1.0
    smf[1] = - cold[1,NJ-1]
    # Non-B.C.
    smp[0,0] = 1.0
    sme[0,1] = -1.0/kappa/ep**1.5 - 1.0/sigma
    smf[0] = -dcdx[0,NJ-1] + (1/kappa/ep**1.5 + 1/sigma)*cold[1,NJ-1] - \
    I/H/W/sigma
    # Insert (sme smp smf) into (smb smd smg)
    smb[:,NJ-1,:] = smp[:,:]
    smd[:,NJ-1,:] = sme[:,:]
    smg[:,NJ-1] = np.transpose(smf)
    return sma, smb, smd, smg

def ABDGXY(sma, smb, smd, smg):
    ( N, NJ )=smg.shape
    sma = np.transpose(sma, (0, 2, 1))
    smb = np.transpose(smb, (0, 2, 1))
    smd = np.transpose(smd, (0, 2, 1))
    A = sma-h/2.0*smb
    B = -2.0*sma+h**2*smd
    D = sma+h/2.0*smb
    G = h**2*smg
    # Old version
    B[:,:,0] = h*smd[:,:,0]-1.5*smb[:,:,0]
    D[:,:,0] = 2.0*smb[:,:,0]
    G[:,0]=h*smg[:,0]
    X = -0.5*smb[:,:,0]
    # Old version
    A[:,:,NJ-1]=-2.0*smb[:,:,NJ-1]
    B[:,:,NJ-1]=h*smd[:,:,NJ-1]+1.5*smb[:,:,NJ-1]
    G[:,NJ-1]=h*smg[:,NJ-1]
    Y=0.5*smb[:,:,NJ-1]
    ABD = np.concatenate((A, B, D), axis=1)

    BC1 = np.concatenate((B[:,:,0] , D[:,:,0] , X), axis=1)
    BC2 = np.concatenate((Y , A[:,:,NJ-1] , B[:,:,NJ-1]), axis=1)
    ABD[:,:,0] = BC1
    ABD[:,:,NJ-1] = BC2
    return ABD, G

def band(ABD, G):
    BMrow = np.reshape(np.arange(1,N*NJ+1), (NJ,N))
    BMrow = BMrow[:, :, np.newaxis]
    BMrow = np.transpose(BMrow, (1, 2, 0))
    BMrow = BMrow[:,[0 for i in range(3*N)],:]
    a = np.arange(1,3*N+1)
    a = a[np.newaxis,:]
    a = np.repeat(a,N,0)
    a = a[:,:,np.newaxis]
    a = np.repeat(a,NJ,2)
    b = np.arange(0,(N)*(NJ-3)+N,N)
    b = np.hstack((b[0], b, b[len(b)-1]))
    b = b[np.newaxis,np.newaxis,:]
    b = np.repeat(b,N,0)
    b = np.repeat(b,3*N,1)
    BMcol = a + b
    BMcol = BMcol - 1
    BMrow = BMrow - 1
    BMrow = np.ravel(BMrow)
    BMcol = np.ravel(BMcol)

    ABD = np.ravel(ABD)
    BigMat = coo_matrix((ABD, (BMrow, BMcol)), shape=(N*NJ, N*NJ)).tocsc()
    BigG = np.transpose(G)
    BigG = np.ravel(BigG)
    delc = spsolve(BigMat, BigG)
    delc = delc.reshape((NJ, N))
    delc = np.transpose(delc)
    return delc

def bound_val1(cold1):
    for iter in range(1,itmax):
        ( dcdx , d2cdx2 )=interpolate( cold1,h)
        (sma,smb,smd,smg)=fillmat1(cold1,dcdx,d2cdx2)
        (ABD,G)=ABDGXY(sma,smb,smd,smg)
        delc= band(ABD,G)
        error=np.amax(np.absolute(delc))
        print ("iter, error = %i, %g" % (iter,error))
        cold1=cold1+delc
        if error < tol:
            return cold1
    print ('The program did not converge!!')

    #disp([h*[1:NJ]',cold',delc'])
    return cold




cold = initguess()
#initial problem solution(which includes overpotential, and ionic current variable)
Sol_init=bound_val1(cold)
#Including time derivative
#1 Time discretization infomation
tstep= 20 #
total_time= 1# 216.6 #h discharge time
totalsteps=int(total_time*3600/tstep)

# Transfer Current
# ===========
def transfer(jcold):
    # Calculates the tranfer current j
    # Uses the 2nd variable in cold, which is i2
    jdcdx, jd2cdx2 = interpolate(jcold,h)
    trans = jdcdx[1,:]
    if abs(trans[1]) <= 1e-3:
        trans[0] = trans[1]
    trans[NJ-1] = trans[NJ-2]
    return trans
# DERIV
# ===========

def deriv(c):
    # Derivative of a 1D array
    cE = np.zeros_like(c)
    cW = np.zeros_like(c)
    cE[0:NJ-1] = c[1:NJ]
    cW[1:NJ] = c[0:NJ-1]
    dd = (cE-cW)/2.0/h
    dd[0] = (-3.0*c[0] + 4.0*c[1] - c[2]) /2.0 /h
    dd[NJ-1] = (3.0*c[NJ-1] - 4.0*c[NJ-2] + c[NJ-3]) /2.0 /h
    return dd
# Electrolyte Properties
# ===========
def electrolyte(concold):
    M = concold[2,:]*1000 # mol/L molarity
    p = (0.99668742) + \
    (5.0065284e-2)*M + \
    (-2.006849e-3)*M**2 + \
    (2.2577517e-4)*M**3 + \
    (-2.276481e-5)*M**4 + \
    (1.3081735e-6)*M**5 + \
    (-3.012443e-8)*M**6 # g/cm3 density

    molal = (4.0074878e-6) + \
    (1.002374947)*M + \
    (7.4141713e-3)*M**2 + \
    (1.3054688e-3)*M**3 + \
    (-1.478333e-5)*M**4 + \
    (2.3063735e-7)*M**5 + \
    (3.562232e-8)*M**6 # mols/kg-H2O molality

    WP = 5610.8*concold[2,:]/p[:] # weight % KOH

    kappainf = (8.782e-3) + \
    (3.720e-2)*WP + \
    (8.109e-5)*WP**2 + \
    (-3.026e-5)*WP**3 + \
    (3.346e-7)*WP**4 # S/cm conductivity

    Dainf = np.exp((-1.0464e1) + \
    (-4.1100e-1)*concold[2,:]**0.5/(1+concold[2,:]**0.5) + \
    (2.9182e-1)*concold[2,:] + \
    (-9.1543e-2)*concold[2,:]**1.5 + \
    (5.9489e-3)*concold[2,:]**2) # cm2/s KOH diff coeff

    gam = np.exp(-(1.1762*molal**0.5)/(1+1.15*molal**0.5) + \
    (0.2302)*molal + \
    (6.0489e-3)*molal**2 + \
    (-2.9934e-4)*molal**3 + \
    (3.9144e-7)*molal**4) # KOH molality activity coeff

    faa = 0.997/(p-56.1056*concold[2,:])*gam # KOH activity coeff
    cwater = (1.0 - concold[2,:]*V_a)/V_0 # mol/cm3 water conc
    return kappainf,Dainf,cwater,faa,WP
# MnO2 Particle Properties
# ===========
def particle(part,jp,tstep):
    # Evolves the MnO2 particle properties
    # 0 is particle radius
    # 1 is inner MnO2 core
    # 2 is lim (proton diffusion across the shell)
    # 3 is aio
    # 4 is X in MnOx
    newpart = np.zeros_like(part)
    ro3 = (-3.0/4.0/3.14/NN/F*(V_alpha-V_gamma)*jp)*tstep + part[0,:]**3
    ro = ro3**(1./3.)
    ro[ro > romax] = romax
    newpart[0,:] = ro
    ri3 = r_ri**3*(newpart[0,:]**3/r_ri**3-V_alpha/V_gamma)/(1.0-V_alpha/V_gamma)
    ri3[ri3 <= 0] = 0
    ri = ri3**(1./3.)
    ri[ri <= 0] = 1.0e-30
    newpart[1,:] = ri
    newpart[2,:] = 4.0*3.14*F*NN*DH/(1.0/newpart[0,:]-1.0/newpart[1,:])
    Apart = 4.0*3.14*newpart[0,:]**2 # cm2 area of MnO2 particle
    a = NN*Apart # cm-1 area per volume MnO2 in cathode
    newpart[3,:] = a*io # A/cm3 exchange transfer current
    newpart[4,:] =((r_ri**3-newpart[0,:]**3)/(V_gamma-V_alpha))*(V_gamma/r_ri**3)
    return newpart
# Anode
# ===========
def anode(anprop,tstep):
    # Evolves the anode properties
    # 0 = anode porosity
    # 1 = convection velocity from anode
    # 2 = anode OH- conc
    # 3 = OH- conc at cathode interface (unchanged here)
    # 4 = overpotential at cathode interface (unchanged here)
    # 5 = OH- flux to anode (unchanged here)
    # 6 = OH- flux from cathode (unchanged here)
    newanprop = np.zeros_like(anprop)
    newanprop[0] = I*(V_Zn - theta*V_ZnO)/2.0/F/V_a*tstep + anprop[0]
    newanprop[1] = -(newanprop[0]-anprop[0])*V_a/Asep/tstep
    newanprop[2] = ((-Asep*anprop[6] - theta*I/F - (1-theta)*2.0*I/F)*tstep/V_a + anprop[2]*anprop[0])/newanprop[0]
    if newanprop[2] <= 0.0:
        newanprop[2] = 0.0
    newanprop[3] = anprop[3]
    newanprop[4] = anprop[4]
    newanprop[5] = anprop[5]
    newanprop[6] = anprop[6]
    newanprop[7] = anprop[7]
    return newanprop
# Fluxes
# ===========
def fluxes(somecold):
    # Calcualtes the OH- flux at each side of something
    kappa,Da,cw,fa,WKOH = electrolyte(somecold)
    dcdr, d2cdr2 = interpolate(somecold,h)
    cE=np.concatenate((somecold[:,1:NJ], np.zeros((N,1)) ), axis=1 )
    cW=np.concatenate(( np.zeros((N,1)), somecold[:,0:NJ-1] ), axis=1 )
    dcdr=(cE-cW)/2.0/h

    dcdr[:,0] = (-3.0*somecold[:,0] + 4.0*somecold[:,1] - somecold[:,2] ) /2.0 /h
    dcdr[:,NJ-1] = (3.0*somecold[:,NJ-1] - 4.0*somecold[:,NJ-2] + somecold[:,NJ-3] )/2.0 /h
    left_flux = -somecold[4,0]**1.5*Da[0]*dcdr[2,0] - somecold[1,0]*t2/F + somecold[2,0]*somecold[3,0]
    right_flux = -somecold[4,NJ-1]**1.5*Da[NJ-1]*dcdr[2,NJ-1] - somecold[1,NJ-1]*t2/F +somecold[2,NJ-1]*somecold[3,NJ-1]
    return left_flux, right_flux

def fillmat(cold,dcdx,d2cdx2,cprev,tstep,part,anprop):
    ( N, NJ )=cold.shape
    sma=np.zeros((N,NJ,N))
    smb=np.zeros((N,NJ,N))
    smd=np.zeros((N,NJ,N))
    smg=np.zeros((N,NJ))
    kappa,Da,cw,fa,WKOH = electrolyte(cold)
    dfa = deriv(fa)
    OHs = anprop[2]
    velo = anprop[1]
    lim = part[2,:]
    aio = part[3,:]
    smb[0,:,0]= -cold[4,:]**1.5*cold[2,:]
    smd[0,:,1]= 1.0/kappa*cold[2,:] + \
    1.0/sigma*cold[2,:]*cold[4,:]**1.5
    smb[0,:,2]= 2.0*R*T/F*(1.0-t2)*cold[4,:]**1.5 + \
    2.0*R*T/F/cw*cold[2,:]*cold[4,:]**1.5
    smd[0,:,2]= 1.0/kappa*cold[1,:] + \
    1.0/sigma*cold[4,:]**1.5*cold[1,:] - \
    I/H/W/sigma*cold[4,:]**1.5 - \
    dcdx[0,:]*cold[4,:]**1.5 + \
    2.0*R*T/F/cw*dcdx[2,:]*cold[4,:]**1.5 + \
    2.0*R*T/F*(1.0-t2)*dfa/fa*cold[4,:]**1.5 + \
    4.0*R*T/F/cw*dfa/fa*cold[4,:]**1.5*cold[2,:]
    smd[0,:,4]= 1.0/sigma*cold[2,:]*cold[1,:]*1.5*cold[4,:]**0.5 - \
    I/H/W/sigma*cold[2,:]*1.5*cold[4,:]**0.5 - \
    dcdx[0,:]*cold[2,:]*1.5*cold[4,:]**0.5 + \
    2.0*R*T/F*(1-t2)*1.5*dcdx[2,:]*cold[4,:]**0.5 + \
    2.0*R*T/F/cw*dcdx[2,:]*1.5*cold[2,:]*cold[4,:]**0.5 + \
    2.0*R*T/F*(1-t2)*dfa/fa*1.5*cold[2,:]*cold[4,:]**0.5 + \
    2.0*R*T/F/cw*dfa/fa*cold[2,:]**2*1.5*cold[4,:]**0.5
    smg[0,:]= + cold[4,:]**1.5*cold[2,:]*dcdx[0,:] - \
    1.0/kappa*cold[1,:]*cold[2,:] - \
    1.0/sigma*cold[2,:]*cold[4,:]**1.5*cold[1,:] + \
    I/H/W/sigma*cold[2,:]*cold[4,:]**1.5 - \
    2.0*R*T/F*(1-t2)*dcdx[2,:]*cold[4,:]**1.5 - \
    2.0*R*T/F/cw*dcdx[2,:]*cold[2,:]*cold[4,:]**1.5 - \
    2.0*R*T/F*(1-t2)*dfa/fa*cold[2,:]*cold[4,:]**1.5 - \
    2.0*R*T/F/cw*dfa/fa*cold[2,:]**2*cold[4,:]**1.5
    smd[1,:,0] = -lim*ba*np.exp(ba*cold[0,:])*cold[2,:] - \
    lim*bc*np.exp(-bc*cold[0,:])*cold[2,:] + \
    1.0*bc*np.exp(-bc*cold[0,:])*dcdx[1,:] + \
    ba*dcdx[1,:]*np.exp(ba*cold[0,:])

    smb[1,:,1] = 1.0/aio*lim*cold[2,:] - \
    1.0*np.exp(-bc*cold[0,:]) + \
    np.exp(ba*cold[0,:])
    smd[1,:,2] = -lim*np.exp(ba*cold[0,:]) + \
    lim*np.exp(-bc*cold[0,:]) + \
    1.0/aio*lim*dcdx[1,:]
    smg[1,:] = lim*cold[2,:]*np.exp(ba*cold[0,:]) - \
    lim*cold[2,:]*np.exp(-bc*cold[0,:]) - \
    lim*dcdx[1,:]*cold[2,:]/aio - \
    dcdx[1,:]*np.exp(ba*cold[0,:]) + \
    dcdx[1,:]*np.exp(-bc*cold[0,:])
    smb[2,:,1] = (t2-1.0)/F
    sma[2,:,2] = Da*cold[4,:]**1.5
    smb[2,:,2] = 1.5*Da*cold[4,:]**0.5*dcdx[4,:] - \
    cold[3,:]
    smd[2,:,2] = - dcdx[3,:] - \
    2.0*cold[4,:]/tstep + \
    cprev[4,:]/tstep
    smb[2,:,3] = -cold[2,:]
    smd[2,:,3] = -dcdx[2,:]
    smb[2,:,4] = 1.5*Da*cold[4,:]**0.5*dcdx[2,:]
    smd[2,:,4] = 1.5*Da*d2cdx2[2,:]*cold[4,:]**0.5 + \
    0.75*Da*cold[4,:]**-0.5*dcdx[2,:]*dcdx[4,:] - \
    2.0*cold[2,:]/tstep + \
    cprev[2,:]/tstep
    smg[2,:] = - Da*d2cdx2[2,:]*cold[4,:]**1.5 - \
    1.5*Da*cold[4,:]**0.5*dcdx[2,:]*dcdx[4,:] + \
    (1-t2)/F*dcdx[1,:]+ \
    cold[3,:]*dcdx[2,:] + \
    cold[2,:]*dcdx[3,:] + \
    2.0*cold[4,:]*cold[2,:]/tstep - \
    cold[4,:]*cprev[2,:]/tstep - \
    cold[2,:]*cprev[4,:]/tstep
    smb[3,:,1] = -(V_0/F+(t2-1.0)*V_a/F)
    smb[3,:,3] = 1.0
    smd[3,:,4] = 1.0/tstep
    smg[3,:] = (V_0/F+(t2-1.0)*V_a/F)*dcdx[1,:]- \
    dcdx[3,:]- \
    cold[4,:]/tstep + \
    cprev[4,:]/tstep
    smb[4,:,1] = -(V_alpha/F - V_gamma/F)
    smd[4,:,4] = 1.0/tstep
    smg[4,:] = -1.0/tstep*cold[4,:] + \
    1.0/tstep*cprev[4,:] + \
    (V_alpha/F - V_gamma/F)*dcdx[1,:]
    #Boundary condition
    smp = np.zeros([N,N])
    sme = np.zeros([N,N])
    smf = np.zeros([N,1])
    sme[1,1] = 1.0
    smf[1] = I/H/W - cold[1,0]
    sme[3,3] = 1.0
    smf[3] = velo-cold[3,0]

    sme[2,2] = 1.0
    smf[2] = OHs - cold[2,0]
    smp[0,0]= -cold[4,0]**1.5*cold[2,0]
    sme[0,1]= 1.0/kappa[0]*cold[2,0] + \
    1.0/sigma*cold[2,0]*cold[4,0]**1.5
    smp[0,2]= 2.0*R*T/F*(1.0-t2)*cold[4,0]**1.5 + \
    2.0*R*T/F/cw[0]*cold[2,0]*cold[4,0]**1.5
    sme[0,2]= 1.0/kappa[0]*cold[1,0] + \
    1.0/sigma*cold[4,0]**1.5*cold[1,0] - \
    I/H/W/sigma*cold[4,0]**1.5 - \
    dcdx[0,0]*cold[4,0]**1.5 + \
    2.0*R*T/F/cw[0]*dcdx[2,0]*cold[4,0]**1.5 + \
    2.0*R*T/F*(1.0-t2)*dfa[0]/fa[0]*cold[4,0]**1.5 + \
    4.0*R*T/F/cw[0]*dfa[0]/fa[0]*cold[4,0]**1.5*cold[2,0]
    sme[0,4]= 1.0/sigma*cold[2,0]*cold[1,0]*1.5*cold[4,0]**0.5 - \
    I/H/W/sigma*cold[2,0]*1.5*cold[4,0]**0.5 - \
    dcdx[0,0]*cold[2,0]*1.5*cold[4,0]**0.5 + \
    2.0*R*T/F*(1-t2)*1.5*dcdx[2,0]*cold[4,0]**0.5 + \
    2.0*R*T/F/cw[0]*dcdx[2,0]*1.5*cold[2,0]*cold[4,0]**0.5 + \
    2.0*R*T/F*(1-t2)*dfa[0]/fa[0]*1.5*cold[2,0]*cold[4,0]**0.5 + \
    2.0*R*T/F/cw[0]*dfa[0]/fa[0]*cold[2,0]**2*1.5*cold[4,0]**0.5
    smf[0]= cold[4,0]**1.5*cold[2,0]*dcdx[0,0] - \
    1.0/kappa[0]*cold[1,0]*cold[2,0] - \
    1.0/sigma*cold[2,0]*cold[4,0]**1.5*cold[1,0] + \
    I/H/W/sigma*cold[2,0]*cold[4,0]**1.5 - \
    2.0*R*T/F*(1-t2)*dcdx[2,0]*cold[4,0]**1.5 - \
    2.0*R*T/F/cw[0]*dcdx[2,0]*cold[2,0]*cold[4,0]**1.5 - \
    2.0*R*T/F*(1-t2)*dfa[0]/fa[0]*cold[2,0]*cold[4,0]**1.5 - \
    2.0*R*T/F/cw[0]*dfa[0]/fa[0]*cold[2,0]**2*cold[4,0]**1.5

    smp[4,1] = -(V_alpha/F - V_gamma/F)
    sme[4,4] = 1.0/tstep
    smf[4] = -1.0/tstep*cold[4,0] + \
    1.0/tstep*cprev[4,0] + \
    (V_alpha/F - V_gamma/F)*dcdx[1,0]
    smb[:,0,:] = smp[:,:]
    smd[:,0,:] = sme[:,:]
    smg[:,0] = np.transpose(smf)
    sme = np.zeros([N,N])
    smp = np.zeros([N,N])
    smf = np.zeros([N,1])
    # B.C.
    sme[1,1] = 1.0
    smf[1]= -cold[1,NJ-1]
    smp[2,2] = 1.0
    smf[2] = -dcdx[2,NJ-1]
    smp[0,0]= -cold[4,NJ-1]**1.5*cold[2,NJ-1]
    sme[0,1]= 1.0/kappa[NJ-1]*cold[2,NJ-1] + \
    1.0/sigma*cold[2,NJ-1]*cold[4,NJ-1]**1.5
    smp[0,2]= 2.0*R*T/F*(1.0-t2)*cold[4,NJ-1]**1.5 + \
    2.0*R*T/F/cw[NJ-1]*cold[2,NJ-1]*cold[4,NJ-1]**1.5
    sme[0,2]= 1.0/kappa[NJ-1]*cold[1,NJ-1] + \
    1.0/sigma*cold[4,NJ-1]**1.5*cold[1,NJ-1] - \
    I/H/W/sigma*cold[4,NJ-1]**1.5 - \
    dcdx[0,NJ-1]*cold[4,NJ-1]**1.5 + \
    2.0*R*T/F/cw[NJ-1]*dcdx[2,NJ-1]*cold[4,NJ-1]**1.5 + \
    2.0*R*T/F*(1.0-t2)*dfa[NJ-1]/fa[NJ-1]*cold[4,NJ-1]**1.5 + \
    4.0*R*T/F/cw[NJ-1]*dfa[NJ-1]/fa[NJ-1]*cold[4,NJ-1]**1.5*cold[2,NJ-1]
    sme[0,4]= 1.0/sigma*cold[2,NJ-1]*cold[1,NJ-1]*1.5*cold[4,NJ-1]**0.5 - \
    I/H/W/sigma*cold[2,NJ-1]*1.5*cold[4,NJ-1]**0.5 - \
    dcdx[0,NJ-1]*cold[2,NJ-1]*1.5*cold[4,NJ-1]**0.5 + \
    2.0*R*T/F*(1-t2)*1.5*dcdx[2,NJ-1]*cold[4,NJ-1]**0.5 + \
    2.0*R*T/F/cw[NJ-1]*dcdx[2,NJ-1]*1.5*cold[2,NJ-1]*cold[4,NJ-1]**0.5 + \
    2.0*R*T/F*(1-t2)*dfa[NJ-1]/fa[NJ-1]*1.5*cold[2,NJ-1]*cold[4,NJ-1]**0.5 + \
    2.0*R*T/F/cw[NJ-1]*dfa[NJ-1]/fa[NJ-1]*cold[2,NJ-1]**2*1.5*cold[4,NJ-1]**0.5
    smf[0]= cold[4,NJ-1]**1.5*cold[2,NJ-1]*dcdx[0,NJ-1] - \
    1.0/kappa[NJ-1]*cold[1,NJ-1]*cold[2,NJ-1] - \
    1.0/sigma*cold[2,NJ-1]*cold[4,NJ-1]**1.5*cold[1,NJ-1] + \
    I/H/W/sigma*cold[2,NJ-1]*cold[4,NJ-1]**1.5 - \
    2.0*R*T/F*(1-t2)*dcdx[2,NJ-1]*cold[4,NJ-1]**1.5 - \
    2.0*R*T/F/cw[NJ-1]*dcdx[2,NJ-1]*cold[2,NJ-1]*cold[4,NJ-1]**1.5 - \
    2.0*R*T/F*(1-t2)*dfa[NJ-1]/fa[NJ-1]*cold[2,NJ-1]*cold[4,NJ-1]**1.5 - \
    2.0*R*T/F/cw[NJ-1]*dfa[NJ-1]/fa[NJ-1]*cold[2,NJ-1]**2*cold[4,NJ-1]**1.5
    smp[3,1] = -(V_0/F+(t2-1.0)*V_a/F)
    smp[3,3] = 1.0
    sme[3,4] = 1.0/tstep
    smf[3] = (V_0/F+(t2-1.0)*V_a/F)*dcdx[1,NJ-1] - \
    dcdx[3,NJ-1] - \
    cold[4,NJ-1]/tstep + \
    cprev[4,NJ-1]/tstep
    smp[4,1] = -(V_alpha/F - V_gamma/F)
    sme[4,4] = 1.0/tstep
    smf[4] = -1.0/tstep*cold[4,NJ-1] + \
    1.0/tstep*cprev[4,NJ-1] + \
    (V_alpha/F - V_gamma/F)*dcdx[1,NJ-1]

    # Insert (sme smp smf) into (smb smd smg)
    smb[:,NJ-1,:] = smp[:,:]
    smd[:,NJ-1,:] = sme[:,:]
    smg[:,NJ-1] = np.transpose(smf)
    return sma, smb, smd, smg

N = Ncath
# Transfer initial result to cprevC
cprevC = np.zeros([N,NJ])
cprevC[0:2,:] = Sol_init[:,:]
cprevC[2,:] = initKOH
cprevC[3,:] = 0
cprevC[4,:] = ep
# Expand coldC for time-dependent problem
coldC = np.zeros([N,NJ])
coldC[:] = cprevC
j = transfer(coldC)
# make MnO2 particle array
partC = np.zeros([5,NJ])
partC[0,:] = r_ri
partC[1,:] = r_ri
partC = particle(partC,j,tstep)
83
Lflux, Rflux = fluxes(coldC)
# ====== Solve initial ANODE problem
propA = np.zeros([8])
propA[0] = epa
propA[2] = initKOH
propA[3] = coldC[2,0]
propA[4] = coldC[0,0]
propA[5] = Lflux
propA[6] = Lflux
propA = anode(propA,tstep)
def bound_val(cold,cprev,tstep,part,anprop):
    for iter in range(1,itmax):
        dcdx, d2cdx2 = interpolate(cold,h)
        sma,smb,smd,smg = fillmat(cold,dcdx,d2cdx2,cprev,tstep,part,anprop)
        ABD,G = ABDGXY(sma,smb,smd,smg)
        delc = band(ABD,G)
        error = np.amax(np.absolute(delc))
        print ("iter, error = %i, %g" % (iter,error))
        cold=cold+delc
        84
        if error < tol:
            return cold
    print ('The program did not converge!!')
    return cold

#solve subsequent problem

#Values need to draw discharge curve
E_0 = 1.34 #V voltage when half of one electron release
Q = m_cathode*0.65*0.308 #A*h theoritical capacity of duracell AA at I = 0.1
for step in range(1,totalsteps+1):
    print
    print ('TIME STEP:', step)
    print ('cathode')
    coldC = bound_val(coldC,cprevC,tstep,partC,propA)
    j = transfer(coldC)
    partC = particle(partC,j,tstep)
    fluxC, junk = fluxes(coldC)
    propA[3] = coldC[2,0]
    propA[4] = coldC[0,0]
    propA[6] = fluxC
    propA = anode(propA,tstep)
    times= step*tstep/3600
    f=I*times/Q
    plt.figure(1)
    85
    plt.plot(times,partC[4,0],'ro')
    if step == 2783:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=5%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=5%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=5%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=5%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=5%')
        plt.legend()
    if step == 5567:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=10%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=10%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=10%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=10%')
        86
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=10%')
        plt.legend()
    if step == 11135:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=20%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=20%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=20%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=20%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=20%')
        plt.legend()
    if step == 16703:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=30%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=30%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=30%')
        plt.legend()

        plt.figure(5)
        plt.plot(rr,-j,label='DOD=30%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=30%')
        plt.legend()
    if step == 22271:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=40%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=40%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=40%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=40%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=40%')
        plt.legend()
    if step == 27838:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=50%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=50%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=50%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=50%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=50%')
        plt.legend()
    if step == 33406:
        plt.figure(2)
        plt.plot(rr,coldC[2,:]*1000,label='DOD=60%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=60%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=60%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=60%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=60%')
        plt.legend()
    if step == 38974:
        plt.figure(2)

        plt.plot(rr,coldC[2,:]*1000,label='DOD=70%')
        plt.legend()
        plt.figure(3)
        plt.plot(rr,coldC[4,:],label='DOD=70%')
        plt.legend()
        plt.figure(4)
        plt.plot(rr,coldC[0,:],label='DOD=70%')
        plt.legend()
        plt.figure(5)
        plt.plot(rr,-j,label='DOD=70%')
        plt.legend()
        plt.figure(6)
        plt.plot(rr,partC[4,:],label='DOD=70%')
        plt.legend()
    cprevC = coldC
plt.figure(1)
plt.xlabel(r'$times(h)$',size=10)
plt.ylabel(r'$y$',size = 10)
plt.ylim(0.0023,0.0225)
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('y20.png',dpi=300)
plt.figure(2)

plt.xlabel(r'$z(cm)$',size = 10)
plt.ylabel(r'$Ca(M)$',size = 10)
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('Ca20.png',dpi = 300)
plt.figure(3)
plt.xlabel(r'$z(cm)$',size = 10)
plt.ylabel(r'$porosity$',size = 10)
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('porosity20.png',dpi = 300)
plt.figure(4)
plt.xlabel(r'$z(cm)$',size = 10)
plt.ylabel(r'$overpotential(V)$',size = 10)
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('ETA920.png',dpi = 300)
plt.figure(5)
plt.xlabel(r'$z(cm)$',size = 10)
plt.ylabel(r'$-j(A/cm^3)$',size = 10)
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('j920.png',dpi = 300)
plt.figure(6)
xxx = np.array([0,0.4])
yyy = np.array([0.79,0.79])

plt.plot(xxx,yyy,'r--')
plt.xlabel(r'$z(cm)$',size = 10)
plt.ylabel(r'$y$',size = 10)
plt.yticks([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True,
right=True)
plt.savefig('xr920.png',dpi = 300)
print(m_cathode)
