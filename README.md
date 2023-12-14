# Dataset Distillation via Adversarial Prediction Matching


This repo contains official code for the paper "[Dataset Distillation via Adversarial Prediction Matching]( )". 


### Getting Started

First, create the conda virtual enviroment

```bash
conda env create -f enviroment.yaml
```

You can then activate your  conda environment with
```bash
conda activate distillation
```

### Obtaining well-trained teachers to be matched
Before doing any distillation, you'll need to generate some teacher models on the original dataset by ```./model_train/train.py```

The following command will train 100 ConvNet models on CIFAR-10 with ZCA whitening for 100 epochs each:
```bash
python ./model_train/train.py --dataset=CIFAR10 --zca --model=ConvNet \
--train_epochs=100 --num_experts=100 --buffer_path=./model_train/models \
--data_path=your_data_path
```
There is an example in the ```./template.sh```


### Adversarial Prediction Matching 
The following command will then use the well-trained teachers we just generated to distill CIFAR-10 down to just 50 image per class:
```bash
CUDA_VISIBLE_DEVICES=6 \
python distill.py --model ConvNet --dataset CIFAR10 --zca --loss l1 --eval_mode ccc  \
--lr_img 1 --ipc 50 --s_epoch 250 --lr_net 0.02 --epoch_eval_train 1000 --num_eval 5 --soft_lab --mid_gap 2 --ce 0.1 \
--data_path=your_data_path
```

More examples for distilling other datasets with different IPCs can be found in the ```./template.sh```

Please use the following hyper-parameters to attain our results reported in Table 1 of the paper:
<img src='docs/parameters.png' width=600>
