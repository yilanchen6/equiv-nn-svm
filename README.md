# Codes for the paper: On the Equivalence between Neural Network andSupport Vector Machine

## Required environments:
`pytorch`     
`neural-tangents`


## Codes
* train_sgd.py: train the NN and SVM with NTK with stochastic subgradient descent.
* config/svm_sgd.yaml: configurations and hyper-parameters to train NN and SVM.
* regression.py: kernel ridge regression with NTK.
* robust_svm:
    * test.py: test the robustness of NN using IBP and SVM with our method in the paper.  
    * test_regressions: test the robustness of kernel ridge regression models using our method.
    
* ibp.py: functions to calculate IBP bounds. Specified for NTK parameterization.
