 
 # LNPL 
 A Model  for learning under noisy and semi-supervised settings
 
 This is the source code for the paper: Murtadha, Ahmed, et al. "Towards Robust Learning with Noisy and Pseudo Labels for Text Classification". If you use the code,  please cite the paper: 
 ```
```
 

# Data



The datasets used in our experminents can be downloaded from this [link](https://drive.google.com/drive/folders/130pP318SQhL8RKBcuHMY_29owiaqbxOm?usp=sharing). 

# Prerequisites:
Required packages are listed in the requirements.txt file:

```
```
# How to use

*  Go to code/         
*  Run the following code to train under noisy label settuings:
```
python run_LNLTC.py --dataset='AG' --noise_percentage=0.2
```

- The params could be :
  - --dataset =\{AG,yelp, yahoo\}
  - --noise_percentage ={0.2, 0.5,...., 0.9},  the ratio of symmatric noise

The results will be written into results/main_lnl.txt

* Run the following code to train under  semi-supervised settuings
```
python run_SSTC.py --dataset='AG' --train_sample=30
```
- The params could be :
   - --dataset =\{AG,yelp, yahoo, TREC,SST, SST-5, CR, MR\}
   - --train_sample ={0, 30,1000, 10000}, where 0 denotes 10% of the labeled data

The results will be written into results/main_sstc.txt

