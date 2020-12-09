# CAN-pytorch
This is a PyTorch implementation of the CAN model described in the paper:
PyTorch version for CAN: Co-embedding Attributed Networks based on code <https://github.com/zfjsail/gae-pytorch> and <https://github.com/mengzaiqiao/CAN> 
>Zaiqiao Meng, Shangsong Liang, Hongyan Bao, Xiangliang Zhang. Co-embedding Attributed Networks. (WSDM2019)


### Requirements
- Python 3.7.4
- PyTorch 1.5.0
- install requirements via 

```
pip install -r requirements.txt
``` 

### How to run

```
python train.py
```


#### Facebook dataset with the default parameter settings

```
Epoch: 0194 train_loss= 0.75375 log_lik= 0.69382 KL= 0.05993 train_acc= 0.73382 val_edge_roc= 0.98516 val_edge_ap= 0.98455 val_attr_roc= 0.95473 val_attr_ap= 0.95721 time= 1.71142
Epoch: 0195 train_loss= 0.75215 log_lik= 0.69217 KL= 0.05998 train_acc= 0.73484 val_edge_roc= 0.98577 val_edge_ap= 0.98492 val_attr_roc= 0.95465 val_attr_ap= 0.95746 time= 1.64731
Epoch: 0196 train_loss= 0.75135 log_lik= 0.69133 KL= 0.06002 train_acc= 0.73486 val_edge_roc= 0.98588 val_edge_ap= 0.98486 val_attr_roc= 0.95322 val_attr_ap= 0.95755 time= 1.64199
Epoch: 0197 train_loss= 0.75140 log_lik= 0.69134 KL= 0.06006 train_acc= 0.73556 val_edge_roc= 0.98545 val_edge_ap= 0.98477 val_attr_roc= 0.95652 val_attr_ap= 0.95914 time= 1.63010
Epoch: 0198 train_loss= 0.75157 log_lik= 0.69146 KL= 0.06010 train_acc= 0.73477 val_edge_roc= 0.98573 val_edge_ap= 0.98490 val_attr_roc= 0.95497 val_attr_ap= 0.95753 time= 1.65039
Epoch: 0199 train_loss= 0.75122 log_lik= 0.69107 KL= 0.06015 train_acc= 0.73400 val_edge_roc= 0.98620 val_edge_ap= 0.98523 val_attr_roc= 0.95420 val_attr_ap= 0.95829 time= 1.66717
Epoch: 0200 train_loss= 0.74931 log_lik= 0.68914 KL= 0.06017 train_acc= 0.73667 val_edge_roc= 0.98601 val_edge_ap= 0.98515 val_attr_roc= 0.95426 val_attr_ap= 0.95744 time= 1.65484
Optimization Finished!
Test edge ROC score: 0.9853779088016957
Test edge AP score: 0.9836879718079673
Test attr ROC score: 0.9578314765862058
Test attr AP score: 0.9577498373032282
```

#### CiteSeer dataset with the default parameter settings  
```
Epoch: 0198 train_loss= 0.81845 log_lik= 0.76834 KL= 0.05011 train_acc= 0.66264 val_edge_roc= 0.94756 val_edge_ap= 0.95467 val_attr_roc= 0.92974 val_attr_ap= 0.92059 time= 1.70837
/Users/storen/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:1558: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
/Users/storen/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:1558: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
Optimization Finished!
Test edge ROC score: 0.9490130318561492
Test edge AP score: 0.95856792990438
Test attr ROC score: 0.9239066109625775
Test attr AP score: 0.9121636661521142
```

#### Cora dataset with the default parameter settings 
```
9027 val_attr_roc= 0.93274 val_attr_ap= 0.91553 time= 0.38018
Epoch: 0192 train_loss= 0.84325 log_lik= 0.76477 KL= 0.07848 train_acc= 0.64674 val_edge_roc= 0.98919 val_edge_ap= 0.99029 val_attr_roc= 0.93418 val_attr_ap= 0.91732 time= 0.36496
Epoch: 0193 train_loss= 0.84597 log_lik= 0.76714 KL= 0.07883 train_acc= 0.65462 val_edge_roc= 0.98916 val_edge_ap= 0.99043 val_attr_roc= 0.93214 val_attr_ap= 0.91432 time= 0.36390
Epoch: 0194 train_loss= 0.84857 log_lik= 0.77013 KL= 0.07844 train_acc= 0.64059 val_edge_roc= 0.98945 val_edge_ap= 0.99055 val_attr_roc= 0.93397 val_attr_ap= 0.91702 time= 0.38282
Epoch: 0195 train_loss= 0.85211 log_lik= 0.77336 KL= 0.07875 train_acc= 0.65224 val_edge_roc= 0.98910 val_edge_ap= 0.99032 val_attr_roc= 0.93098 val_attr_ap= 0.91294 time= 0.36883
Epoch: 0196 train_loss= 0.84762 log_lik= 0.76950 KL= 0.07812 train_acc= 0.64099 val_edge_roc= 0.98931 val_edge_ap= 0.99041 val_attr_roc= 0.93387 val_attr_ap= 0.91710 time= 0.36784
Epoch: 0197 train_loss= 0.84100 log_lik= 0.76280 KL= 0.07819 train_acc= 0.65151 val_edge_roc= 0.98937 val_edge_ap= 0.99063 val_attr_roc= 0.93326 val_attr_ap= 0.91600 time= 0.35659
Epoch: 0198 train_loss= 0.83551 log_lik= 0.75756 KL= 0.07795 train_acc= 0.65085 val_edge_roc= 0.98941 val_edge_ap= 0.99061 val_attr_roc= 0.93529 val_attr_ap= 0.91857 time= 0.36861
Epoch: 0199 train_loss= 0.83763 log_lik= 0.75975 KL= 0.07788 train_acc= 0.64954 val_edge_roc= 0.98940 val_edge_ap= 0.99060 val_attr_roc= 0.93542 val_attr_ap= 0.91888 time= 0.36662
Epoch: 0200 train_loss= 0.83960 log_lik= 0.76150 KL= 0.07810 train_acc= 0.65307 val_edge_roc= 0.98948 val_edge_ap= 0.99075 val_attr_roc= 0.93391 val_attr_ap= 0.91660 time= 0.37454
Optimization Finished!
Test edge ROC score: 0.9890442712427842
Test edge AP score: 0.9865145173317201
Test attr ROC score: 0.9420356776264327
Test attr AP score: 0.9279907464117463
```
