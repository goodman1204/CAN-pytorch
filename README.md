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



#### Cora dataset with the default parameter settings 
```
Epoch: 0188 train_loss= 0.86054 log_lik_u= 0.42103 log_lik_a= 0.36872 KL_u= 0.03137 KL_a= 0.03941 train_acc= 0.64043 val_edge_roc= 0.98926 val_edge_ap= 0.98999 val_attr_roc= 0.93950 val_attr_ap= 0.92674 time= 0.36615
Epoch: 0189 train_loss= 0.85163 log_lik_u= 0.43425 log_lik_a= 0.34606 KL_u= 0.03150 KL_a= 0.03983 train_acc= 0.64813 val_edge_roc= 0.98957 val_edge_ap= 0.99030 val_attr_roc= 0.94228 val_attr_ap= 0.92948 time= 0.38405
Epoch: 0190 train_loss= 0.85577 log_lik_u= 0.44019 log_lik_a= 0.34403 KL_u= 0.03157 KL_a= 0.03997 train_acc= 0.64338 val_edge_roc= 0.99000 val_edge_ap= 0.99069 val_attr_roc= 0.94106 val_attr_ap= 0.92807 time= 0.38525
Epoch: 0191 train_loss= 0.85074 log_lik_u= 0.42411 log_lik_a= 0.35529 KL_u= 0.03150 KL_a= 0.03985 train_acc= 0.64297 val_edge_roc= 0.98993 val_edge_ap= 0.99073 val_attr_roc= 0.94236 val_attr_ap= 0.92941 time= 0.39881
Epoch: 0192 train_loss= 0.85172 log_lik_u= 0.42165 log_lik_a= 0.35852 KL_u= 0.03159 KL_a= 0.03996 train_acc= 0.64434 val_edge_roc= 0.98939 val_edge_ap= 0.99029 val_attr_roc= 0.94171 val_attr_ap= 0.92841 time= 0.36417
Epoch: 0193 train_loss= 0.84897 log_lik_u= 0.43346 log_lik_a= 0.34335 KL_u= 0.03187 KL_a= 0.04029 train_acc= 0.64905 val_edge_roc= 0.98940 val_edge_ap= 0.99041 val_attr_roc= 0.94193 val_attr_ap= 0.92811 time= 0.38592
Epoch: 0194 train_loss= 0.84761 log_lik_u= 0.43347 log_lik_a= 0.34161 KL_u= 0.03202 KL_a= 0.04050 train_acc= 0.64833 val_edge_roc= 0.98985 val_edge_ap= 0.99072 val_attr_roc= 0.94173 val_attr_ap= 0.92784 time= 0.33626
Epoch: 0195 train_loss= 0.84754 log_lik_u= 0.42390 log_lik_a= 0.35094 KL_u= 0.03206 KL_a= 0.04064 train_acc= 0.64433 val_edge_roc= 0.98986 val_edge_ap= 0.99063 val_attr_roc= 0.94298 val_attr_ap= 0.92965 time= 0.36720
Epoch: 0196 train_loss= 0.84444 log_lik_u= 0.42264 log_lik_a= 0.34853 KL_u= 0.03226 KL_a= 0.04101 train_acc= 0.65003 val_edge_roc= 0.98936 val_edge_ap= 0.99012 val_attr_roc= 0.94336 val_attr_ap= 0.93031 time= 0.36503
Epoch: 0197 train_loss= 0.84725 log_lik_u= 0.43198 log_lik_a= 0.34125 KL_u= 0.03257 KL_a= 0.04146 train_acc= 0.65535 val_edge_roc= 0.98884 val_edge_ap= 0.98960 val_attr_roc= 0.94336 val_attr_ap= 0.93008 time= 0.33206
Epoch: 0198 train_loss= 0.84265 log_lik_u= 0.42680 log_lik_a= 0.34164 KL_u= 0.03269 KL_a= 0.04152 train_acc= 0.65313 val_edge_roc= 0.98944 val_edge_ap= 0.99011 val_attr_roc= 0.94389 val_attr_ap= 0.93048 time= 0.38881
Epoch: 0199 train_loss= 0.84453 log_lik_u= 0.42224 log_lik_a= 0.34794 KL_u= 0.03281 KL_a= 0.04154 train_acc= 0.64681 val_edge_roc= 0.99014 val_edge_ap= 0.99072 val_attr_roc= 0.94386 val_attr_ap= 0.93045 time= 0.38584
Epoch: 0200 train_loss= 0.84126 log_lik_u= 0.42577 log_lik_a= 0.34060 KL_u= 0.03309 KL_a= 0.04181 train_acc= 0.65237 val_edge_roc= 0.99010 val_edge_ap= 0.99091 val_attr_roc= 0.94382 val_attr_ap= 0.93027 time= 0.38437
Optimization Finished!
Test edge ROC score: 0.9874611980862964
Test edge AP score: 0.9850135333312862
Test attr ROC score: 0.9418874300102977
Test attr AP score: 0.9273113354286826
```
