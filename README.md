# GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm)

![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/banner-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/banner-1.jpg)
###### GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Banner #1

***


## Current Status
[![contributor for this repository](https://img.shields.io/badge/contributor-rubenandrebarreiro-blue.svg)](https://github.com/rubenandrebarreiro/) [![developed in](https://img.shields.io/badge/developed&nbsp;in-fct&nbsp;nova-blue.svg)](https://www.fct.unl.pt/)
[![current version](https://img.shields.io/badge/version-1.0-magenta.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)

[![status of this version no. 1](https://img.shields.io/badge/status-completed-orange.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![status of this version no. 2](https://img.shields.io/badge/status-final-orange.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![status of this version no. 3](https://img.shields.io/badge/status-stable-orange.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![status of this version no. 4](https://img.shields.io/badge/status-documented-orange.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)

[![keyword of this version no. 1](https://img.shields.io/badge/keyword-high&nbsp;performance&nbsp;computing-brown.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![keyword of this version no. 2](https://img.shields.io/badge/keyword-neural&nbsp;networks-brown.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![keyword of this version no. 3](https://img.shields.io/badge/keyword-som-brown.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)
[![keyword of this version no. 4](https://img.shields.io/badge/keyword-cuda-brown.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)

[![technology used no. 1](https://img.shields.io/badge/built&nbsp;with-cuda-red.svg)](https://developer.nvidia.com/cuda-zone) 
[![technology used no. 2](https://img.shields.io/badge/built&nbsp;with-c++-red.svg)](http://www.cplusplus.com/) 
[![technology used no. 3](https://img.shields.io/badge/built&nbsp;with-c-red.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![technology used no. 4](https://img.shields.io/badge/built&nbsp;with-cmake-red.svg)](https://cmake.org/) 
[![software used no. 1](https://img.shields.io/badge/software-jetbrains&nbsp;clion-gold.svg)](https://www.jetbrains.com/clion/)

[![star this repository](http://githubbadges.com/star.svg?user=rubenandrebarreiro&repo=gpu-cuda-self-organising-maps&style=flat)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/stargazers)
[![fork this repository](http://githubbadges.com/fork.svg?user=rubenandrebarreiro&repo=gpu-cuda-self-organising-maps&style=flat)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/fork)
[![downloads of this repository](https://img.shields.io/github/downloads/rubenandrebarreiro/gpu-cuda-self-organising-maps/total.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/archive/master.zip)
[![price of this project](https://img.shields.io/badge/price-free-success.svg)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/archive/master.zip)

##### Current Progress of the Project

[![current progress of this project](http://progressed.io/bar/100?title=&nbsp;completed&nbsp;)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/) 

##### Final Approval Grade of the Project

[![grade of this project](http://progressed.io/bar/0?scale=20&title=&nbsp;grade&nbsp;&suffix=&nbsp;)](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/)


## Description

> A [**_1st year's lab work (project)_**](http://www.unl.pt/guia/2018/fct/UNLGI_getCurso?curso=935) of the [**_MSc. degree of Computer Science and Engineering_**](https://www.fct.unl.pt/en/education/course/integrated-master-computer-science/) made in [**_FCT NOVA (Faculty of Sciences and Technology of New University of Lisbon)_**](https://www.fct.unl.pt/), in the subject of [**_High Performance Computing_**](http://www.unl.pt/guia/2018/fct/UNLGI_getUC?uc=11165). This project was built using [**_CUDA (Compute Unified Device Architecture)_**](https://developer.nvidia.com/cuda-zone), [**_C++ (C Plus Plus)_**](http://www.cplusplus.com/), [**_C_**](https://en.wikipedia.org/wiki/C_(programming_language)), [**_CMake_**](https://cmake.org/) and [**_JetBrains CLion_**](https://www.jetbrains.com/clion/). The scenario of the project was a **_GPU-based_** implementation of the [**_Self-Organising-Maps (S.O.M.) algorithm_**](https://en.wikipedia.org/wiki/Self-organizing_map) for [**_Artificial Neural Networks (A.N.N.)_**](https://en.wikipedia.org/wiki/Artificial_neural_network), with the support of [**_CUDA (Compute Unified Device Architecture)_**](https://developer.nvidia.com/cuda-zone), using its offered _parallel optimisations_ and _tunings_. The final goal of the project was to test the several **_GPU-based_** implementations of the algorithm against a given **_CPU-based_** implementation of the same algorithm and, _evaluate_ and _compare_ the _overall performance_ (_speedup_, _efficiency_ and _cost_).

[Click here to see the Project's Description!](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/project-description/PDFs/project-description-1.pdf)
######  GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Project's Description #1

***

### 1) Goal

> The goal of this project is to develop a **_GPU's implementation of Self-Organising Map (SOM) algorithm_**, and compare its performance (_speedup_, _efficiency_ and _cost_) against a given _sequential implementation_ of the algorithm.

> You may implement your solution in either **_CUDA_** or **_OpenCL_**.

> No higher level frameworks, such as **_Thrust_**, **_SkePU_**, **_TensorFlow_**, **_Marrow_**, or others, are allowed.

> The project must be carried out by a group of, at most, 2 students.


### 2) Self-Organising Map

> **_S.O.M._** is a very popular artificial neural network model that is trained via unsupervised learning, meaning that the learning process does not require human intervention and that not much needs to be known about the data itself.

> The algorithm is used for clustering (feature detection) and visualization in exploratory data analysis. Application fields include pattern recognition, data mining and process optimization.

> If you are curious about the fundamentals and the application of the algorithm, you can check the following site, which is a good starting point:

* [https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms](https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms)

> Note however, that you do not need this information to accomplish the task asked in this project.

> The _SOM algorithm_ is presented in _Algorithm #1_. The fitting of the model to the input dataset is represented by a map (represented by variable _map_), which is an array of _nrows_ ∗ _ncols_ vectors of _nfeatures_ features, i.e., a tensor of size _nrows × ncols × nfeatures_. The algorithm begins by initialing map with vectors of random values, and then, for each input i performs the following steps:

1. Compute the distance from _i_ to all vectors of _map_. The _distance_ function may be any. You will be asked to implement 2 functions.

2. Compute the Best Matching Unit (_bmu_), which is the vector closest (with minimum distance) to _i_. Note that the _argmin_ function returns the coordinate of the map where the _bmu_ is.

3. Update the map, given the input _i_ and the _bmu_.

With regard to the update map procedure, several learning rates may be considered. In this project you will consider only the one given by formula:

```
learning rate(t) = 1/t
```

> **_Algorithm #1_**

![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/algorithm-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/algorithm-1.jpg)
######  GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Algorithm #1


### 3) GPU Implementation

> You must implement a _C/C++ program_ that receives the following command line:

```
gpu_som number_rows number_columns datafile outputfile [distance]
```

> _where **number_rows** and **number_cols** denote, respectively, the number of rows and columns of the map, **datafile** is the name of the file holding the input data set, and **outputfile** is the name of the file to where the final state of the map must
be written. **distance** is the parameter that allows the user to choose between the two distance functions to implement. It is a optional parameter defaulted to the Euclidean distance._

> Given this configuration, your program must execute in the GPU as much of the presented _SOM algorithm_ as possible. In particular, the _SOM_ map must reside is GPU’s memory and be modified there, as it receives inputs read from the input file and transferred to the GPU. A close analysis to _Algorithm #1_ will unveil several massively parallel computations, such as the ones that are performed for all the vectors of the matrix.

> The map must only be transferred to the host explicitly via a function implemented for the purpose, with a name such as get map. You must naturally do this at the end of the computation, to store the map’s final state to the output file, but you can use it for debugging purposes (not in the version to evaluate for performance).

> In order for you to test and evaluate your solution, several input data files will be provided in the next few days.

### 4) Requirements

#### a) Distance Functions

> As mentioned in the previous section, you must implement two distance functions:

##### i) Euclidean 2D Distance (mandatory)
![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/euclidean-2d-distance-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/euclidean-2d-distance-1.jpg)

##### ii) Cosine Distance (mandatory)
![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/cosine-distance-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/cosine-distance-1.jpg)

##### iii) Manhattan Distance (bonus)
![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/manhattan-distance-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/manhattan-distance-1.jpg)

##### iv) Minkowski Distance (bonus)
![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/minkowski-distance-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/minkowski-distance-1.jpg)

##### v) Chebyshev Distance (bonus)
![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/chebyshev-distance-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/distances-functions/chebyshev-distance-1.jpg)


#### b) Performance Measurements

> You must calculate the execution time of the versions (one version for each distance function) of your algorithm from the moment the SOM map is initialized (do not include this initialization) up until the map is written to the output file. These execution times must be compared against a sequential version that will be given in the next few days.

#### c) Report

> Along with the code of your solution, you must deliver a report of, at most, 5 pages presenting your solution, your experiment results and your conclusions. Concerning the solution, focus on explaining which parts of the algorithm are executed on the
GPU, and describing the algorithms you devised to accomplish such execution.

##### i) Implementation's report

[Click here to see the Project's Report!](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/report/PDFs/report-1.pdf)
######  GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Report #1

***

## Screenshots

![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/screenshot-1.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/screenshot-1.jpg)
######  GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Screenshot #1

***

![https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/screenshot-2.jpg](https://raw.githubusercontent.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/master/imgs/JPGs/screenshot-2.jpg)
######  GPU's CUDA - Self-Organising Maps (S.O.M. Algorithm) - Screenshot #2

***


## Getting Started

### Prerequisites
To install and run this application, you will need:
> The [**_Git_**](https://git-scm.com/) feature and/or a [**_third-party Git Client based GUI_**](https://git-scm.com/downloads/guis/), like:
* [**_GitHub Desktop_**](https://desktop.github.com/), [**_GitKraken_**](https://www.gitkraken.com/), [**_SourceTree_**](https://www.sourcetreeapp.com/) or [**_TortoiseGit_**](https://tortoisegit.org/).

### Installation
To install this application, you will only need to _download_ or _clone_ this repository and run the application locally:

> You can do it downloading the [**_.zip file_**](https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps/archive/master.zip) in download section of this repository.

> Or instead, by cloning this repository by a [**_Git Client based GUI_**](https://git-scm.com/downloads/guis), using [**_HTTPS_**](https://en.wikipedia.org/wiki/HTTPS) or [**_SSH_**](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol), by one of the following link:
* [**_HTTPS_**](https://en.wikipedia.org/wiki/HTTPS):
```
https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps.git
```
* [**_SSH_**](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol):
```
git@github.com:rubenandrebarreiro/gpu-cuda-self-organising-maps.git
```

> Or even, by running one of the following commands in a **_Git Bash Console_**:
* [**_HTTPS_**](https://en.wikipedia.org/wiki/HTTPS):
```
git clone https://github.com/rubenandrebarreiro/gpu-cuda-self-organising-maps.git
```
* [**_SSH_**](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol):
```
git clone git@github.com:rubenandrebarreiro/gpu-cuda-self-organising-maps.git
```

## Built with
* [**_CUDA (Compute Unified Device Architecture)_**](https://developer.nvidia.com/cuda-zone)
* [**_C++ (C Plus Plus)_**](http://www.cplusplus.com/)
* [**_C_**](https://en.wikipedia.org/wiki/C_(programming_language))
* [**_CMake_**](https://cmake.org/)
* [**_JetBrains CLion_**](https://www.jetbrains.com/clion/)


## Contributors

> [Rúben André Barreiro](https://github.com/rubenandrebarreiro/)

## Contacts

### Rúben André Barreiro
#### E-mails
* [ruben.barreiro.92@gmail.com](mailto:ruben.barreiro.92@gmail.com)
* [r.barreiro@campus.fct.unl.pt](mailto:r.barreiro@campus.fct.unl.pt)
* [up201808917@fe.up.pt](mailto:up201808917@fe.up.pt)
* [up201808917@g.uporto.pt](mailto:up201808917@g.uporto.pt)

## Portfolios/Blogs and Git Hosting/Repository Services

### Rúben André Barreiro
#### GitHub's Portfolio/Personal Blog
* [https://rubenandrebarreiro.github.io/](https://rubenandrebarreiro.github.io/)

#### Hosting/Repository Services
* [https://github.com/rubenandrebarreiro/](https://github.com/rubenandrebarreiro/)
* [https://gitlab.com/rubenandrebarreiro/](https://gitlab.com/rubenandrebarreiro/)
* [https://bitbucket.org/rubenandrebarreiro/](https://bitbucket.org/rubenandrebarreiro/)
* [https://dev.azure.com/rubenandrebarreiro/](https://dev.azure.com/rubenandrebarreiro/)

## Classifications/Final Approval Grades

### Approval Grade of Lab Work
* 18.5 of 20
