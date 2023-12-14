CUDA_VISIBLE_DEVICES=6 \
python ./model_train/train.py --dataset=CIFAR10 --zca --model=ConvNet \
--train_epochs=100 --num_experts=100 --buffer_path=./model_train/models \
--data_path=your_data_path

CUDA_VISIBLE_DEVICES=6 \
python distill.py --model ConvNet --dataset CIFAR10 --zca --loss l1 --eval_mode ccc  \
--lr_img 1 --ipc 50 --s_epoch 250 --lr_net 0.02 --epoch_eval_train 1000 --num_eval 5 --soft_lab --mid_gap 2 --ce 0.1 \
--data_path=your_data_path

CUDA_VISIBLE_DEVICES=6 \
python distill.py --model ConvNet --dataset CIFAR100 --zca --loss l1 --eval_mode ccc  \
--lr_img 1 --ipc 10 --s_epoch 250 --lr_net 0.02 --epoch_eval_train 1000 --num_eval 5 --soft_lab --mid_gap 2 --ce 0.1 \
--data_path=your_data_path

CUDA_VISIBLE_DEVICES=6 \
python distill.py --model ConvNetD4 --dataset Tiny --loss l1 --eval_mode ccc  \
--lr_img 20 --ipc 10 --s_epoch 250 --lr_net 0.1 --epoch_eval_train 1000 --num_eval 5 --soft_lab --mid_gap 2 --ce 0.1 \
--batch_syn 500 --update_syn 100 --data_path=your_data_path