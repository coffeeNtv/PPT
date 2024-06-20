import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    print("iter_path",iter_path)
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)


    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0 # the total number of training iterations

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d ' % (start_epoch))        
    else:    
        start_epoch, epoch_iter = 1, 0

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()      # timer for entire epoch
        iter_data_time = time.time()        # timer for data loading per iteration
        epoch_iter = 0                      # the number of training iterations in current epoch, reset to 0 every epoch

        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):           # inner loop within one epoch
            epoch_iter = epoch_iter % dataset_size  
            iter_start_time = time.time()            # timer for computation per iteration
            data['current_epoch'] = epoch       
            data['current_iter'] = total_steps
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset() # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if epoch == start_epoch and i == 0:
                model.data_dependent_initialize(data)
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                temp_visuals=model.get_current_visuals()
                visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            iter_data_time = time.time()

        # end of epoch 
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))  
            model.save('latest')  
            model.save(epoch)
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
