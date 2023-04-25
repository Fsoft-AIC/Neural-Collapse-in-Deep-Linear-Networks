# Neural Collapse in Deep Linear Networks: From Balanced to Imbalanced Data

This is the code for the [paper](https://arxiv.org/abs/2301.00437) "Neural Collapse in Deep Linear Networks: From Balanced to Imbalanced Data".

International Conference on Machine Learning (ICML), 2023
## Experiments
***Balanced data***\
Multilayer perceptron experiment\
args: --depth_linear in [1, 3, 6, 9]; --width in [512, 1028, 2048]; --no-bias for bias-free (leave empty for last-layer bias )
```
CUDA_VISIBLE_DEVICES=0 python train_1st_order.py --model MLP --dataset cifar10 \
            --depth_relu 6 --depth_linear 1 --width 512 --seed 1 \
            --loss MSE --lr 0.0001 --optimizer Adam --sep_decay 
CUDA_VISIBLE_DEVICES=0 python validate_NC.py --model MLP --dataset cifar10 \
            --depth_relu 6 --depth_linear 1 --width 512 --seed 1 \
            --loss MSE --lr 0.0001 --optimizer Adam --sep_decay 
```
Deep learning experiment\
args: --model in [ResNet18, VGG16], --width in [512], --depth_linear in [1, 3, 6, 9]
```
CUDA_VISIBLE_DEVICES=0 python train_1st_order.py --model ResNet18 --dataset cifar10 \
--depth_linear 1 --width 512 --seed 1
CUDA_VISIBLE_DEVICES=0 python validate_NC.py --model ResNet18 --dataset cifar10 \
--depth_linear 1 --width 512 --seed 1
```
Direct optimization experiment\
args: --no-bias for bias-free (leave empty for last-layer bias )
``` 
CUDA_VISIBLE_DEVICES=0 python synthetic_experiment_balanced.py --hidden 64 --num_iteration 30000 --lr 0.1
```
***Imbalanced data***\
Multilayer perceptron experiment\
args: --width in [512], --depth_linear in [1, 3, 6]
```
CUDA_VISIBLE_DEVICES=0 python train_1st_order_unbalanced.py --model MLP --dataset cifar10 \
            --depth_relu 6 --depth_linear 1 --width 2048 --seed 1 --no-bias \
            --loss MSE --lr 0.0001 --weight_decay 0.00001 --optimizer Adam --sep_decay  --epochs 12000 --patience 6000
CUDA_VISIBLE_DEVICES=0 python validate_NC_unbalanced.py --model MLP --dataset cifar10 \
            --depth_relu 6 --depth_linear 1 --width 2048 --seed 1 --no-bias \
            --loss MSE --lr 0.0001 --weight_decay 0.00001 --optimizer Adam --sep_decay  --epochs 12000 --patience 6000
```
Direct optimization experiment\
args: --width in [1, 3, 6], --depth_linear in [1, 3, 6, 9]
```
CUDA_VISIBLE_DEVICES=0 python synthetic_experiment_imbalanced.py --hidden 2048 --num_iteration 30000 --depth_linear 1
```
## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2301.00437).
```
@article{dang2023neural,
  title={Neural Collapse in Deep Linear Network: From Balanced to Imbalanced Data},
  author={Hien Dang and Tho Tran and Stanley Osher and Hung Tran-The and Nhat Ho and Tan Nguyen},
  journal={arXiv preprint arXiv:2301.00437},
  year={2023}
}
```