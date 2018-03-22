# Advertisement Prediction

discrete data preprocessing: one hot encode~
CTR (Click-through rate) prediction: Logistic Regression, Factorization Machines (FM)

## Feature-cross method

* FNN: Factorization-machine supported Neural Network
* PNN : Product-based NN 
* Wide& Deep: LR+NN
* DeepFM : FM + DNN

## Multi-Task Learning

* Hard & Soft Parameter Sharing
* DRN: Deep Relationship Network
* Cross-Stitch Network

Loss in Multi-Task Learning:

* Loss: loss1 + loss2 + ...
* Improved Loss: loss1 + Y * loss2 ...  (derived from possibility model)

[AUC?](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it) 

## Experiment

* Baseline 1 : $X-Y_2$, predict $p(Y_2=1)$

* Baseline 2 : $X-Y_1$, predict $p(Y_1=1)$

  ​		$X(Y_1=1)-Y_2$, predict $p(Y_2=1|Y_1=1)$

  ​