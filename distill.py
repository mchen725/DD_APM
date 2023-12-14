''''
This is the formal implementation of "Dataset Distillation via Adversarial Prediction Matching" framework
'''

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

import tqdm
import torch.nn.functional as F


def load_teacher(args):    
    model_teacher = get_network(args.model, 3, args.num_classes, args.im_size, dist=False) # get a random model

    expert_path = './model_train/models/'+args.dataset+'/'+args.model+'/'
    expert_files = os.listdir(expert_path)
    rand_id1 = np.random.choice(len(expert_files))
    state = torch.load(expert_path + expert_files[rand_id1])
    model_teacher.load_state_dict(state)
    if args.distributed:
        model_teacher = nn.DataParallel(model_teacher)
    model_teacher.to(args.device)
    
    model_teacher.eval()

    return model_teacher
    

def train(it_eval, net, images_train, labels_train, args, middle_models, texture=False):
    net = net.to(args.device)
    images_train = images_train
    labels_train = labels_train
    lr = float(args.lr_net)
    Epoch = args.s_epoch
    # lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,[Epoch//2+1],gamma=0.1)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == 'l1':
        criterion = F.l1_loss

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    save_eps = list(np.arange(0,Epoch, Epoch//args.s_num))
    save_eps = save_eps[1:]+[Epoch]

    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        # if ep in save_eps or ep==Epoch:
        if ep in save_eps:
            middle_models.append(copy.deepcopy(net).cpu())
        
        lr_schedule.step()


    time_train = time.time() - start
    if it_eval % 10==0:
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train))

    return middle_models


def evaluate_synset_mc(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    # lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,[Epoch//2+1],gamma=0.1)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == 'l1':
        criterion = F.l1_loss

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
        # if ep % 200 ==0 or ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, nn.CrossEntropyLoss().to(args.device), args, aug=False)
            time_train = time.time() - start
            print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

        lr_schedule.step()

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test

def get_lab(net,images_train,label,args):
    dst_train = TensorDataset(images_train,label)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=32, shuffle=False, num_workers=0)
    preds = []
    for idx, datam in enumerate(trainloader):
        img = datam[0].to(args.device)
        pred = net(img)
        preds.append(pred.detach().cpu())
    preds = torch.concat(preds,dim=0)
    return preds

def divide_chunks(lst, n):
    """Divide a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='ccc', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=2500, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--img_mom', type=float, default=0.5, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='num synthetic data involved in one iteration')
    parser.add_argument('--update_syn', type=int, default=None, help='num synthetic data for update')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--soft_lab', action='store_true')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--zca', action='store_true', help='zca whitening')
    parser.add_argument('--loss', type=str, default='l1', choices=['ce','l1'], help='type')
    parser.add_argument('--s_epoch', type=int, default=500, help='epoch num for training surrogate model, refers to E in the paper')
    parser.add_argument('--s_num', type=int, default=5, help='num of student models stored during the training, refer to K in the paper')
    parser.add_argument('--mid_gap', type=int, default=5, help='Switch the proxy models every mid_gap iterations.')
    parser.add_argument('--ce', type=float, default=10, help='coefficient for ce loss')



    args = parser.parse_args()
    args.method = 'distill_data'
    args.distributed = torch.cuda.device_count() > 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 50).tolist()[1:]
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, _, _, _ = get_dataset(args.dataset, args.data_path, args=args)
    args.num_classes = num_classes
    args.im_size = im_size
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    if args.update_syn is None:
        args.update_syn = args.batch_syn


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device='cpu')

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device='cpu')
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device='cpu').view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        label_syn_ori = copy.deepcopy(label_syn)
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        
        ce_criterion = nn.CrossEntropyLoss().to(args.device)
        # print('########## test teacher model ##########')
        # teacher_model = load_teacher(args)
        # _, acc_test = epoch('test', testloader, teacher_model, None, ce_criterion, args, aug=False)
        # print('teach test acc:', acc_test)

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=args.img_mom) # optimizer_img for synthetic data
        lr_img_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_img, milestones=[args.Iteration//2], gamma=0.3)
        # optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img, betas=[0.5, 0.9], eps = 1e-8)
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())
        softmax = nn.Softmax()
        
        cur_best_acc = 0
        count = args.mid_gap
        avg_time = 0
        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                teacher_model = load_teacher(args)
                image_syn_eval = copy.deepcopy(image_syn.detach().cpu())
                label_syn = get_lab(teacher_model,image_syn_eval,label_syn_ori,args)
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu()) # avoid any unaware modification
           
                        _, acc_train, acc_test = evaluate_synset_mc(it, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        # print('acc_test', acc_test)

                    cur_avg_acc = np.mean(accs)
                    if cur_avg_acc > cur_best_acc and model_eval==args.model:
                        print('saving......')
                        cur_best_acc = cur_avg_acc
                        if args.save:
                            data_save = [image_syn_eval, label_syn_eval]
                            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'gap%s_ce%s_%s_%s_%dipc_init%s_loss%s_lrimg%s_sepoch%s_ev%s_mom%s_lrnet%s_best.pt'%(args.mid_gap,args.ce, args.dataset, args.model, args.ipc, args.init,args.loss,args.lr_img, args.s_epoch, args.epoch_eval_train, args.img_mom, args.lr_net)))
                    # if it == eval_it_pool[-1] and args.save:
                    #     data_save = [image_syn_eval, label_syn_eval]
                    #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'gap%s_ce%s_%s_%s_%dipc_init%s_loss%s_lrimg%s_sepoch%s_ev%s_mom%s_lrnet%s_final.pt'%(args.mid_gap,args.ce, args.dataset, args.model, args.ipc, args.init,args.loss,args.lr_img, args.s_epoch, args.epoch_eval_train, args.img_mom, args.lr_net)))
                    
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    print('current test best acc::', cur_best_acc)
                    print('args:',args)

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs


            ''' Train a model from scratch for E steps on synthetic data, and save K checkpoints. '''
            start_time = time.time()
            indices = torch.randperm(len(image_syn))[:args.batch_syn]
            chunks = list(divide_chunks(indices, args.update_syn))
            image_syn_cur = image_syn[indices]
            label_syn_cur = label_syn_ori[indices]

            if count == args.mid_gap:
                teacher_model = load_teacher(args)
                teacher_model.eval()
                count = 0
                middle_models = []
                net = get_network(args.model, channel, num_classes, im_size) # get a random model
                image_syn_cur = copy.deepcopy(image_syn_cur.detach())

                teach_all_logits = get_lab(teacher_model,image_syn_cur,label_syn_cur,args)
                # print('teach_all_logits shape',teach_all_logits.size())
                if args.loss == 'ce':
                    label_syn = softmax(teach_all_logits).detach()
                elif args.loss == 'l1':
                    label_syn = teach_all_logits.detach()
                    
                middle_models = train(it, net, image_syn_cur, label_syn, args, middle_models)

            else:
                count += 1


            loss_all = 0
            accumulated_grad = torch.zeros_like(image_syn)
            
            for n in range(len(middle_models)):             
                cur_net = middle_models[n].to(args.device)
                cur_net.eval()
                # print('n',n)
                for chunk in chunks:
                    if len(chunk)==1:
                        img = image_syn[chunk].to(args.device).unsqueeze(0)
                    else:
                        img = image_syn[chunk].to(args.device)
                    t_logits = teacher_model(img)
                    s_logits = cur_net(img)
                    
                    # we omit the logarithmic as it will not impact the optimization
                    loss_l1 = - (len(t_logits)/args.batch_syn) * F.l1_loss( s_logits, t_logits ) 
                    loss_all += loss_l1.data.detach()
                    loss_l1.backward(retain_graph=False)
                    
                    accumulated_grad += image_syn.grad.detach()
                    image_syn.grad.zero_()  # reset the gradient
       

            ''' Squentially update each sub-group within current mini-batch of synthetic samples for flexible time and memory tradeoff '''
            loss_tce = 0
            for chunk in chunks:
                if len(chunk)==1:
                    img = image_syn[chunk].to(args.device).unsqueeze(0)
                    lab = label_syn_ori[chunk].to(args.device).unsqueeze(0)
                else:
                    img = image_syn[chunk].to(args.device)
                    lab = label_syn_ori[chunk].to(args.device)
                t_logits = teacher_model(img)
                loss_tce = (len(t_logits)/args.batch_syn) * args.ce * ce_criterion(t_logits, lab)
                loss_tce.backward(retain_graph=False)
                accumulated_grad += image_syn.grad.detach()
                image_syn.grad.zero_()  # reset the gradient
            image_syn.grad = accumulated_grad

            optimizer_img.step()
            lr_img_scheduler.step()
            optimizer_img.zero_grad()
            avg_time += time.time()-start_time
            
            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_all))
                print('avg time/iter:',avg_time/(it+1))
                
      

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()


