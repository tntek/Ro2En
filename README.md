# Motion Planning Networks
Code (pytorch) for  [Robust Environment Encoder for Zero-shot Fast Motion Planning]() on C3D, C3D3, C3D5, C3D8, C3D10.

## Overview of Ro2En-based motion planning

![](/results/fig3.png)

## Data Download

You need to download the following data.

* [**C3D**](https://drive.google.com/file/d/1wNPfdVGkkZ-7haTUhdzT0sGnZAkAJEol/view?usp=sharing)
* [**C3D3**, **C3D5**,**C3D8**,**C3D10**](https://drive.google.com/drive/folders/1aDuwkiYG6lfHbQ10J2vp-bfh9-2gZlK6?usp=sharing)

## Python  Dependencies

- python==3.6.13

- open3d==0.13.0

- numpy==1.19.5

- pytorch==1.10.1

## Training and evaluation


**Ro2En:**

```shell
python MPNet/AE/Ro2En.py	//train Ro2En
python MPNet/train.py	//train neural planner
python 3Dtest.py	// evaluation on target domains
python MPNet/neuralplanner.py //evaluation on source domain
```

**MPNet:**

```shell
python MPNet/AE/CAE.py	//train MPNet_CAE
python MPNet/train.py	//train neural planner
python 3Dtest.py	// evaluation on target domains
python MPNet/neuralplanner.py //evaluation on source domain 
```
*Note*: 
* Before training the neural planner, you need to set the environment encoder to use in [data_loader.py](/MPNet/data_loader.py).
* When testing you need to determine which environment encoder and neural planner you want to use in [3Dtest.py](3Dtest.py).
## Results

![](/results/results.png)


## Acknowledgement

The code is based on [MPNet](https://github.com/ahq1993/MPNet).

## Contact
- tntechlab@hotmail.com
