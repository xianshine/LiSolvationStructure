#!/usr/bin/python
#--------------------------------------------------------------------------------------------
# code used to analyze solvation environment of lithium ion in different electrolytes
# for the paper 
# Molecular design for electrolyte solvents enabling energy-dense and long-cycling lithium metal batteries
# Zhiao, Yu; Hansen, Wang; Xian, Kong; William, Huang; Yuchi, Tsao; David G., Mackanic; Kecheng, Wang; Xinchang, Wang; Wenxiao, Huang; Snehashis, Choudhury; Yu, Zheng; Chibueze V., Amanchukwu; Samantha T., Hung; Yuting, Ma; Eder G., Lomeli; Jian, Qin; Yi, Cui; Zhenan, Bao
# Nature Energy, 2020, 5(7): 526-533.
# DOI: 10.1038/s41560-020-0634-5
# the default trajectory and MD setting are rdf.xtc and rdf.tpr, respectively
# For questions, reach Xian Kong at xianshine@gmail.com
#--------------------------------------------------------------------------------------------


import math as m
import os
import numpy as np
import argparse

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import TPR, XTC

#Natural Constants
kB=1.38064852e-23
e=1.60217662e-19
T=300.0
beta=1.0/(kB*T)
dr=0.001

#process inputs
parser = argparse.ArgumentParser()
parser.add_argument("-ani",help="anion", default='TFS')
parser.add_argument("-sol",help="solvent", default='EC')
parser.add_argument("-b",help="begin time (ps)", default=0,type=float)
parser.add_argument("-dt",help="time step in trajectory (ps)", default=1,type=float)
parser.add_argument("-dta",help="time step for analysis (ps)", default=10,type=float)
args = parser.parse_args()

ani=args.ani
sol=args.sol
beg=args.b
dt=args.dt
dta=args.dta

sols=[ani,sol]

if sol=='ED':
   sols=[ani,'EC','DEC']
if sol=='EDF':
   sols=[ani,'EC','DEC','FEC']
if sol=='DB':
   sols=[ani,'DMC','BTF']
if sol=='FD':
   sols=[ani,'FEC','DEC']
if sol=='FFD':
   sols=[ani,'FEC','FEM','D2E']

atmani="O"
if ani=='PF6':
    atmani="F"
if ani=='BF4':
    atmani="F"
if ani=='I':
    atmani="I"

nsols=len(sols)

maxSol=10
maxAni=10

if nsols==4:
   counts=np.zeros([maxAni,maxSol,maxSol,maxSol])
elif nsols==3:
   counts=np.zeros([maxAni,maxSol,maxSol])
elif nsols==2:
   counts=np.zeros([maxAni,maxSol])
else:
   print("unknown number of solvents, exiting...")
   exit()

#--------------------------------------------------------------
# setting cutoff value, in A, mannually, for testing
#--------------------------------------------------------------

rcuts=3.0*np.ones(nsols)

#--------------------------------------------------------------
# determing cutoff value from RDFs, used in the manuscript
#--------------------------------------------------------------

#rcuts=np.zeros(nsols)
#frdfLiN='rdf.LI.{:s}of{:s}.xvg'.format(atmani,ani)
#dat0=np.fromfile(frdfLiN,sep=" ")
#print( len(dat0) )
#rdfLiN=dat0.reshape(int(len(dat0)/2),2)
#iMax=np.argmax(rdfLiN[0:int(0.6/dr),1])
#iMin=np.argmin(rdfLiN[iMax:int(0.6/dr),1])
#rcuts[0]=10*rdfLiN[iMax+iMin,0]

#for isol in range(1,nsols):
#   si=sols[isol]
#   frdfLiOE='rdf.LI.OEof'+si+'.xvg'
#   frdfLiOC='rdf.LI.OCof'+si+'.xvg'
#   if os.path.isfile(frdfLiOC):
#      frdfLiO=frdfLiOC
#   else:
#      frdfLiO=frdfLiOE

#   dat0=np.fromfile(frdfLiO,sep=" ")
#   rdfLiO=dat0.reshape(int(len(dat0)/2),2)
#   iMax=np.argmax(rdfLiO[:,1])
#   iMin=np.argmin(rdfLiO[iMax:int(0.6/dr),1])
#   rcuts[isol]=10*rdfLiO[iMax+iMin,0]

#--------------------------------------------------------------
# Note: For results in the paper, we choose the minimum of all the rcuts 
# in above line as the cut-off for determining first solvation shell
# This can be done by adding this line:
# rcuts[0:]=np.amin(rcuts[0:])
#--------------------------------------------------------------

print(rcuts)

u = mda.Universe("rdf.tpr", "rdf.xtc", in_memory=False)
li = u.select_atoms("resname LI")
#print(u.atoms)
#lit=li[0]
#print(lit.resid)

nFrame = len(u.trajectory)

#print(nFrame)
#print (resIDSol)
#print(li.atoms)

matRes=np.zeros((nFrame,1))

for ts in u.trajectory[int(beg/dt):]:
   if (u.trajectory.time > beg) and (u.trajectory.time %dta==0 ):
      #print("Frame: {0:5d}, Time: {1:8.3f} ps, nsol: {2:5d} ".format(ts.frame, u.trajectory.time, 0))
      for a in li.atoms:
         agLi1=mda.core.groups.AtomGroup([a])
         nSolRes=np.zeros((nsols,), dtype=int)
         for isol in range(0,nsols):
            si=sols[isol]
            #solAtms = u.select_atoms("resname {0:5s} and name OC OE N3 NBT B Cl1 P  I and around {1:8.3f} group li1".format(si,rcuts[isol]),li1=agLi1)
            # For the results in the paper, we didn't consider the situtation of binding by F atoms. Later we found that it is necessary to include F atoms in FDMB
            #solAtms = u.select_atoms("resname {0:5s} and (not name C* B* P* H* S* F*) and around {1:8.3f} group li1".format(si,rcuts[isol]),li1=agLi1)
            # This is for considering the binding with F atoms in FDMB
            solAtms = u.select_atoms("resname {0:5s} and (not name C* B* P* H* S*) and around {1:8.3f} group li1".format(si,rcuts[isol]),li1=agLi1)
            solRes = solAtms.residues
            resIDSol = solRes.resids
            #print(solAtms)
            #print(solRes)
            #print(resIDSol)
            #if (isol==0) and (len(resIDSol)==5):
            #   ids=' '.join(['{:10d}'.format(i) for i in resIDSol])
            #   print("Time: {:8.0f} ps; Li resID: {:10d}".format(u.trajectory.time,agLi1[0].resid))
            #   print(ids)
            #   exit()
            nSolRes[isol] = len(resIDSol)
            #print("Frame: {0:5d}, Time: {1:8.3f} ps, nsol: {2:5d} ".format(ts.frame, u.trajectory.time, nSolRes[isol]))
         if nsols==4:
            counts[nSolRes[0],nSolRes[1],nSolRes[2],nSolRes[3]] += 1.0
         elif nsols==3:
            counts[nSolRes[0],nSolRes[1],nSolRes[2]] += 1.0
         elif nsols==2:
            counts[nSolRes[0],nSolRes[1]] += 1.0

#print(counts)
sumCounts=np.sum(counts)
#print(sumCounts)
freqs=counts/sumCounts*100

fo=open('coordNumFreq2.log',"w")

if nsols==4:
 for i1 in range(maxAni):
  for i2 in range(maxSol):
   for i3 in range(maxSol):
    for i4 in range(maxSol):
     if freqs[i1,i2,i3,i4]>=1.0:
      print("{:5d} {:5d} {:5d} {:5d} {:8.3f}".format(i1,i2,i3,i4,freqs[i1,i2,i3,i4]) )
      fo.write("{:5d} {:5d} {:5d} {:5d} {:8.3f} \n".format(i1,i2,i3,i4,freqs[i1,i2,i3,i4]) )
elif nsols==3:
 for i1 in range(maxAni):
  for i2 in range(maxSol):
   for i3 in range(maxSol):
     if freqs[i1,i2,i3]>=1.0:
      print("{:5d} {:5d} {:5d} {:8.3f}".format(i1,i2,i3,freqs[i1,i2,i3]) )
      fo.write("{:5d} {:5d} {:5d} {:8.3f} \n".format(i1,i2,i3,freqs[i1,i2,i3]))
elif nsols==2:
 for i1 in range(maxAni):
  for i2 in range(maxSol):
     if freqs[i1,i2]>=1.0:
      print("{:5d} {:5d} {:8.3f}".format(i1,i2,freqs[i1,i2]) )
      fo.write("{:5d} {:5d} {:8.3f} \n".format(i1,i2,freqs[i1,i2]))
fo.close()

#print(counts)
#np.savetxt('rcf.'+sol+'.dat',rcf,fmt='%6.3f %10.6f')
