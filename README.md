# On the Equivalence between Neural Network and Support Vector Machine
Codes for NeurIPS 2021 paper "[On the Equivalence between Neural Network and Support Vector Machine]()".


# Overview
In this paper, we prove the equivalence between neural network (NN) and support vector machine (SVM), specifically, the 
infinitely wide NN trained by soft margin loss and the standard soft margin SVM with NTK trained by subgradient descent. 
Our main theoretical results include establishing the equivalence between NN and a broad family of L2 regularized 
kernel machines (KMs) with finite-width bounds, which cannot be handled by prior work, and showing that every 
finite-width NN trained by such regularized loss functions is approximately a KM. 

Furthermore, we demonstrate our theory can enable three practical applications, including 
- *non-vacuous* generalization bound of NN via the corresponding KM; 
- *non-trivial* robustness certificate for the infinite-width NN (while existing robustness verification methods 
(e.g. IBP, Fast-Lin, CROWN) would provide vacuous bounds); 
- intrinsically more robust infinite-width NNs than those from previous kernel regression.  

See our [paper]() and [slides](http://chenyilan.net/files/SVM_Slides.pdf) for details.

![Equivalence between infinite-width NNs and a family of KMs](https://github.com/leslie-CH/svm/blob/main/examples/table1.png)

# Cite our paper
Yilan Chen, Wei Huang, Lam M. Nguyen, Tsui-Wei Weng, "[On the Equivalence between Neural Network and Support Vector Machine]()", NeurIPS 2021.

```
@inproceedings{chen2021equiv,
title={On the equivalence between neural network and support vector machine},
author={Yilan Chen and Wei Huang and Lam M. Nguyen and Tsui-Wei Weng},
booktitle={Advances in Neural Information Processing Systems},
year={2021}
}
```



# Code overview
* `train_sgd.py`: train the NN and SVM with NTK with stochastic subgradient descent. Plot the results to verify the equivalence.
* `generalization.py`: compute *non-vacuous* generalization bound of NN via the corresponding KM.  
* `regression.py`: kernel ridge regression with NTK.
* `robust_svm.py`:
    * `test()`: evaluate the robustness of NN using IBP or SVM with our method in the paper.  
    * `test_regressions()`: evaluate the robustness of kernel ridge regression models using our method.
    * `bound_ntk()`：calculate the lower and upper bound for NTK of two-layer fully-connected NN.
* `ibp.py`: functions to calculate IBP bounds. Specified for NTK parameterization.

* `models/model.py`: codes for constructing fully-connected neural networks with NTK parameterization.
* `config/`:
  * `svm_sgd.yaml`: configurations and hyper-parameters to train NN and SVM.
  * `svm_gene.yaml`: configurations and hyper-parameters to calculate generalization bound.


# Required environments:
This code is tested on the below environments:
```
Python==3.8.8
torch==1.8.1
neural-tangents==0.3.6
```
For the installation of `PyTorch`, please reference the instructions from https://pytorch.org/get-started/locally/. 
For the installation and usage of `neural-tangents`, please reference the instructions at https://github.com/google/neural-tangents. 



# Experiments
### Train NN and SVM to verify the equivalence
```
python train_sgd.py
```
#### Example of the SGD results
![SGD results](https://github.com/leslie-CH/svm/blob/main/examples/plot_sgd.png)

#### Example of the GD results
![GD results](https://github.com/leslie-CH/svm/blob/main/examples/output.png)


### Computing *non-vacuous* generalization bound of NN via the corresponding KM
```
python generalization.py
```
#### Example of the generalization bound results
![Generalization bound results](https://github.com/leslie-CH/svm/blob/main/examples/generalization.png)



### Train kernel ridge regression with NTK models
```
python regression.py
```

### Robustness verification of NN
Add your paths to your NN models in the code and separate by the width. Specify the width of the models you want to verify.
Then run the `test()` function in `robust_svm`.
```
python -c "import robust_svm; robust_svm.test('nn')"
```


### Robustness verification of SVM
Add your paths to your SVM models in the code. Then run the `test()` function in `robust_svm.py`.
```
python -c "import robust_svm; robust_svm.test('svm')"
```
![robustness verification results](https://github.com/leslie-CH/svm/blob/main/examples/table2.png)



### Robustness verification of kernel regression models
Run `test_regressions()` function in `robust_svm.py`.
```
python -c "import robust_svm; robust_svm.test_regressions()"
```
![robustness verification results](https://github.com/leslie-CH/svm/blob/main/examples/table3.png)
