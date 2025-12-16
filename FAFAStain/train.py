import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
# from torch.utils.tensorboard import SummaryWriter
import os   

if __name__ == '__main__':
    train_start_time = time.time()  # 新增：记录训练开始时间
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt.train_dataset_size = dataset_size
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1
    # tensorboard_log_dir = './checkpoints/train/log'
    # if not os.path.exists(tensorboard_log_dir):
    #     os.makedirs(tensorboard_log_dir)
    # writer = SummaryWriter(tensorboard_log_dir)
    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            data['current_epoch'] = epoch
            data['current_iter'] = total_iters
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data_init = 0)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_time = (time.time() - optimize_start_time)  / batch_size

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                
                #########记录稳定性#########
                # 记录稳定性：例如 generator loss 的方差
                if 'G' in losses:
                    if not hasattr(opt, 'G_loss_history'):
                        opt.G_loss_history = []
                    opt.G_loss_history.append(losses['G'])
                    if len(opt.G_loss_history) >= 50:
                        recent = opt.G_loss_history[-50:]
                        mean_G = sum(recent) / len(recent)
                        var_G = sum((x - mean_G) ** 2 for x in recent) / len(recent)
                        print(f"[Stability Monitor] G Loss (last 50): mean={mean_G:.4f}, var={var_G:.4f}")
                ###########

                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if opt.n_epochs_decay == 0:
            pass
        else:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.

    total_train_time = time.time() - train_start_time
    total_hours = int(total_train_time // 3600)
    total_minutes = int((total_train_time % 3600) // 60)
    total_seconds = int(total_train_time % 60)
    print(f"Total training time: {total_hours}h {total_minutes}m {total_seconds}s")
    with open(os.path.join(opt.checkpoints_dir, opt.name, "train_time_log.txt"), 'a') as f:
        f.write(f"Total training time: {total_hours}h {total_minutes}m {total_seconds}s\n")


