[[Project Page](https://ml-jku.github.io/UPT)]


**U**niversal **P**hysics **T**ransformers (UPTs) are a novel learning paradigm that can model a wide range of 
spatio-temporal problems - both for Lagrangian and Eulerian discretization schemes.


<p align="center">
<img width="100%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/schematic.png">
</p>


UPTs can model transient flow simulations (Eulerian discretization scheme) as indicated by test loss and rollout performance (measured via correlation time):

<p align="center">
<img width="48%" alt="maect_schematic" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_scaling_testloss.svg">
<img width="48%" alt="lowshot_vitl" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_scaling_corrtime.svg">
</p>


<p align="center">
<img width="100%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_rollout.png">
</p>


and also the flow-field of particle based simulations (Lagrangian discretization scheme):

<p align="center">
<img width="100%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/lagrangian_field.png">
</p>