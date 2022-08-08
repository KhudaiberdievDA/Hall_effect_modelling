# This program is to calculate the potential of the Hall bar in the Hall effect regime for any rectangular shape

# The problem with modelling the Hall effect (classically) is that in high magnetic field (or in regime when SigmaXY >> Sigma XX) edge equations become one-dimensional and thus it's not right to tie them on both contacts. Thus, one should break the symmetry which is a bit strange for Finite Element Modelling where all the equations are pretty solid depending on the geometry. 
# Solution that seems correct for me now is to use the edge channels of the QHE regime with the conductivity equals to SigmaXY. One can show that modelling this way and using the standart "bulk" theory give the same result, but the first one is appliable to FEM.
# Thus, I declare the next stept for writting the program

1. Restore the program with any rectangular geometry but no edge chanenls (maybe add the anisotropy)
2. Add the QHE  edge channels with SigmaXY conductivity
3. Add the QSHE edge channels with g and gamma
4. Add the interface 
5. Add some custom geometries so people can chose 