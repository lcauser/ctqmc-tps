# Transition path sampling methods for continuous-time quantum monte carlo
A simple implementation of continuous-time quantum monte carlo, based on local trajectory updates. These updates resample the trajectory of a local spatial degree of freedom while keeping the remainder of the trajectory fixed. This update is exact in the sense it samples the sub-trajectory with a probability consistent with the partition function, making it rejection-free. 

The first application is to the transverse field ising model (TFIM),
```math
\hat{H}^{\rm TFIM} = -\sum_{i} \hat{X}_{i} - J\sum_{\langle i, j \rangle} \hat{Z}_{i}\hat{Z}_{j}.
```
Here, we update the trajectory of a single spin, using the neighbouring spins to model a time-dependant dynamics for the spin which is being updated. Example code for one and two dimensions can be found in the TFIM subdirectory.

The second applicaiton is the quantum triangular plaquette model (QTPM),
  ```math
\hat{H}^{\rm TFIM} = -\sum_{\{i, j, k\}\in\triangle} \hat{X}_{i}\hat{X}_{j}\hat{X}_{k} - J\sum_{i} \hat{Z}_{i}.
```
The update scheme we propose here is to update the trajectory of a single plaquette of spins, $\triangle$. Example code can be found in the QTPM subdirectory.

Detailed explanations of the methods can be found at **insert arXiv link**.
