# One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks

This is the repository of our submission for IEEE Transactions on PAMI   "One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks". 

Here is the preprint vision [TBD]().

## Abstract
Inspired by neuronal diversity in the biological neural system, a plethora of studies proposed to design novel types of artificial neurons and introduce neuronal diversity into artificial neural networks. Recently proposed quadratic neuron, which replaces the inner-product operation in conventional neurons with a quadratic one, have achieved great success in many essential tasks. Despite the promising results of quadratic neurons, there is still an unresolved issue: *Is the superior performance of quadratic networks simply due to the increased parameters or due to the intrinsic expressive capability?* Without clarifying this issue, the performance of quadratic networks is always suspicious. Additionally, resolving this issue is reduced to finding killer applications of quadratic networks. In this paper, \ff{with theoretical and empirical studies, we show that quadratic networks enjoy parametric efficiency, thereby confirming that the superior performance of quadratic networks is due to the intrinsic expressive capability instead of the increased parameters.} Theoretically, we derive the approximation efficiency of the quadratic network in terms of real space and manifolds using Taylor's theorem. We propose two theorems for the error bounds of quadratic networks, suggesting that a large number of parameters are saved in quadratic networks compared to conventional ones. Moreover, from the perspective of the Barron space, we demonstrate that there exists a functional space whose functions can be approximated by quadratic networks in a dimension-free error, but the approximation error of conventional networks is dependent on dimensions. Empirically, we systematically conduct experiments on synthetic data, classic benchmarks, and real-world applications, and compare the parametric efficiency between quadratic models and their state-of-the-art competitors. Experimental results show that quadratic models broadly enjoy parametric efficiency, and the gain of efficiency depends on the task.




All experiments are conducted with Windows 10 on an Intel i9 10900k CPU at 3.70 GHz and one NVIDIA RTX 3080Ti 12GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## Citing
If you find this repo useful for your research, please consider citing it:
```
TBD
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
We conducted separate experiments on the efficiency of second-order networks under different datasets, and the programs were stored in separate folders

### Datasets
* Car experiment: 
* Cell experiment:



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



## Contact
If you have any questions about our work, please contact the following email address:

jingxiaoliao@hit.edu.cn

Enjoy your coding!
## Reference

