# Codes for the paper: On the Equivalence between Neural Network andSupport Vector Machine



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

## Cite our paper
Yilan Chen, Wei Huang, Lam M. Nguyen, Tsui-Wei Weng, "[On the Equivalence between Neural Network and Support Vector Machine]()", NeurIPS 2021.

```
  BibText here
```



# Required environments:
`pytorch`     
`neural-tangents`


# Codes overview
* train_sgd.py: train the NN and SVM with NTK with stochastic subgradient descent.
* config/svm_sgd.yaml: configurations and hyper-parameters to train NN and SVM.
* regression.py: kernel ridge regression with NTK.
* robust_svm:
    * test.py: test the robustness of NN using IBP and SVM with our method in the paper.  
    * test_regressions: test the robustness of kernel ridge regression models using our method.
    
* ibp.py: functions to calculate IBP bounds. Specified for NTK parameterization.


