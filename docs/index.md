

[[Code](https://github.com/ml-jku/UPT)] [[Paper](https://arxiv.org/abs/2402.12365)] [[Talk](https://youtu.be/mfrmCPOn4bs)]  [[Tutorial](https://github.com/BenediktAlkin/upt-tutorial)] [[Codebase Demo Video](https://youtu.be/80kc3hscTTg)] [[BibTeX](https://github.com/ml-jku/UPT#citation)]


**U**niversal **P**hysics **T**ransformers (UPTs) are a novel learning paradigm to efficiently train large-scale
neural operators for a wide range of 
spatio-temporal problems - both for Lagrangian and Eulerian discretization schemes.


<p align="center">
<img width="100%" alt="schematic" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/schematic.png">
</p>


The current landscape of neural operators mainly focuses on small-scale problems using models that do not scale well
to large-scale 3D settings. UPTs compress the (potentially) high-dimensional input into a compressed latent 
space with makes them very scalable. Starting with 32K inputs (scale 1), we scale the number of input points
and evaluate the memory required to train such a model with batchsize 1. In this qualitative scaling study,
UPTs can scale up to 4.2M inputs (scale 128), 64x more than a GNN could handle.


<p align="center">
<img width="80%" alt="scaling" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_limits_comparison.svg">
</p>


The architecture of UPT consists of an encoder, an approximator and a decoder. The encoder is responsible to encode
the physics domain into a latent representation, the approximator propagates the latent representation forward in time
and the decoder transforms the latent representation back to the physics domain.

<p align="center">
<img width="80%" alt="architecture1" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/architecture1.svg">
</p>


To enforce the responsibilities of each component, inverse encoding and decoding tasks are added.


<p align="center">
<img width="80%" alt="architecture2" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/architecture2.svg">
</p>



UPTs can model transient flow simulations (Eulerian discretization scheme) as indicated by test loss and rollout performance (measured via correlation time):

<p align="center">
<img width="48%" alt="cfd_scaling_testloss" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_scaling_testloss.svg">
<img width="48%" alt="cfd_scaling_corrtime" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_scaling_corrtime.svg">
</p>


<p align="center">
<img width="100%" alt="cfd_rollout" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/cfd_rollout.png">
</p>


UPTs can also model the flow-field of particle based simulations (Lagrangian discretization scheme):

<p align="center">
<img width="100%" alt="lagrangian_field" src="https://raw.githubusercontent.com/ml-jku/UPT/main/.github/imgs/lagrangian_field.png">
</p>
Particles show the ground truth velocities of particles and the white arrows show the learned velocity field of a UPT model evaluated on the positions of a regular grid.