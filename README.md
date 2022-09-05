 #### LNPL:Towards Robust Learning with Noisy and Pseudo Label for Text Classification


The datasets used in our experminents can be downloaded from this [link](https://drive.google.com/drive/folders/130pP318SQhL8RKBcuHMY_29owiaqbxOm?usp=sharing). 

### To train the LNLP under learn with noisy labels, use code/run_LNLTC.py
```
python run_LNLTC.py --dataset='AG' --noise_percentage=0.2
```

#### The params could be :
#### --dataset =\{AG,yelp, yahoo\}
#### --noise_percentage ={0.2, 0.5,...., 0.9}
The results will be written into results/main_lnl.txt

### To train the LNLP under learn with noisy labels, use code/run_SSTC.py
```
python run_SSTC.py --dataset='AG' --train_sample=30
```
### The params are:
#### --dataset =\{AG,yelp, yahoo, TREC,SST, SST-5, CR, MR\}
#### --train_sample ={0, 30,1000, 10000}, where 0 denotes 10% of the labeled data

The results will be written into results/main_sstc.txt

