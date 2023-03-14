# One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks

This is the repository of our submission for IEEE Transactions on PAMI   "One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks". 

Here is our preprint version [arxiv](https://arxiv.org/pdf/2303.06316.pdf).


## Abstract
Inspired by neuronal diversity in the biological neural system, a plethora of studies proposed to design novel types of artificial neurons and introduce neuronal diversity into artificial neural networks. Recently proposed quadratic neuron, which replaces the inner-product operation in conventional neurons with a quadratic one, have achieved great success in many essential tasks. Despite the promising results of quadratic neurons, there is still an unresolved issue: *Is the superior performance of quadratic networks simply due to the increased parameters or due to the intrinsic expressive capability?* Without clarifying this issue, the performance of quadratic networks is always suspicious. Additionally, resolving this issue is reduced to finding killer applications of quadratic networks. In this paper, \ff{with theoretical and empirical studies, we show that quadratic networks enjoy parametric efficiency, thereby confirming that the superior performance of quadratic networks is due to the intrinsic expressive capability instead of the increased parameters.} Theoretically, we derive the approximation efficiency of the quadratic network in terms of real space and manifolds using Taylor's theorem. We propose two theorems for the error bounds of quadratic networks, suggesting that a large number of parameters are saved in quadratic networks compared to conventional ones. Moreover, from the perspective of the Barron space, we demonstrate that there exists a functional space whose functions can be approximated by quadratic networks in a dimension-free error, but the approximation error of conventional networks is dependent on dimensions. Empirically, we systematically conduct experiments on synthetic data, classic benchmarks, and real-world applications, and compare the parametric efficiency between quadratic models and their state-of-the-art competitors. Experimental results show that quadratic models broadly enjoy parametric efficiency, and the gain of efficiency depends on the task.




All experiments are conducted with Windows 10 on an Intel i9 10900k CPU at 3.70 GHz and one NVIDIA RTX 3080Ti 12GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## Citing
If you find this repo useful for your research, please consider citing it:
```
@misc{https://doi.org/10.48550/arxiv.2303.06316,
  doi = {10.48550/ARXIV.2303.06316},
  url = {https://arxiv.org/abs/2303.06316},
  author = {Fan, Feng-Lei and Dong, Hang-Cheng and Wu, Zhongming and Ruan, Lecheng and Zeng, Tieyong and Cui, Yiming and Liao, Jing-Xiao},
  title = {One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

```


## Repository organization

### Requirements
We use PyCharm 2021.2 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.8
* PyTorch == 1.10.1
* CUDA == 11.3 if use GPU
* wandb == 0.12.11
* anaconda == 2021.05
 
### Organization
We conducted separate experiments on the efficiency of second-order networks under different datasets, and the programs were stored in separate folders.

The code of **car experiment** we refer to this [Github repo](https://github.com/milesial/Pytorch-UNet).


The code of **cell experiment** we refer to this [Github repo](https://github.com/Andy-zhujunwen/UNET-ZOO).

The code of **bearing experiment** we refer to this [Github repo](https://github.com/asdvfghg/QCNN_for_bearing_diagnosis).

### Datasets
* Car experiment: We use Carvana image masking challenge [1] as the evaluation  dataset. The dataset is available in [Here](https://github.com/abhijitkulkarni25/Kaggle-Carvana-Dataset).
* Cell experiment: We use 2018 Data Science Bowl (DSB2018) [2] as this evaluation dataset. The data is shared in [Here](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
* Bearing experiment: We use our own data for bearing fault diagnosis. For legislative reasons, we cannot share our data. But you can use some public dataset, such as CWRU [3] for evaluation. 


### How to Use
#### 1. wandb
Some models use **Weight & Bias** for training and fine-tuning. It is the machine learning platform for developers to build better models faster. [Here](https://docs.wandb.ai/quickstart) is a quick start for Weight & Bias. You need to create an account and install the CLI and Python library for interacting with the Weights and Biases API:
```
pip install wandb
```
Then login 
```
wandb login
```
#### 2. Training and Testing
In the subfolder, the main program entry is in the 'main.py' or 'train.py'.
 


## Main Results
### Classification Performance
Because we have done many types of experiments, please see our paper for results in details.


## Contact
If you have any questions about our work, please contact the following email address:

jingxiaoliao@hit.edu.cn

Enjoy your coding!
## Reference
[1] https://www.kaggle.com/c/carvana-image-masking-challenge

[2] https://www.kaggle.com/competitions/data-science-bowl-2018/overview

[3] https://engineering.case.edu/bearingdatacenter/welcome
