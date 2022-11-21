import glob
import wandb
import os

from base import *
from eval import val_result
from loss_funcs import Loss, Learnable_Loss
np.set_printoptions(precision=2, suppress=True)
from lib.dataset.pw3d import PW3D
from torch.autograd import Variable

dataset_dict = {'pw3d':PW3D}

wandb.login()

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self._build_model_()
        self._build_optimizer()
        self.set_up_val_loader()
        self._calc_loss = Loss()
        self.loader = self._create_data_loader(train_flag=True)
        self.mutli_task_uncertainty_weighted_loss = Learnable_Loss(self.loader.dataset._get_ID_num_()).cuda()
        self.optimizer.add_param_group({'params': self.mutli_task_uncertainty_weighted_loss.parameters()})
        
        self.train_cfg = {'mode':'matching_gts', 'is_training':True, 'update_data': True, 'calc_loss': True if self.model_return_loss else False, \
                           'new_training': args().new_training}
        self.val_best_PAMPJPE = {'pw3d': 60}
        logging.info('Initialization of Trainer finished!')


    def train(self):
        #init_seeds(self.local_rank, cuda_deterministic=False)
        if args().use_wandb and args().local_rank==0:
            path = "/z/home/mkhoshle/Human_object_transform/wandb"
            wandb.init(project=args().dataset, dir=path, name=args().exp, entity='mkhoshle', config=args)
            
            #Replace result dir with wandb unique id, much easier to find checkpoints
            run_id = wandb.run.id 

            result_dir = os.path.join(args().save_dir, '_'.join((args().dataset, run_id)))
            log_dir    = os.path.join(result_dir, 'logs')
            save_dir   = os.path.join(result_dir, 'checkpoints')
            
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(log_dir,    exist_ok=True) 
            os.makedirs(save_dir,   exist_ok=True)
        
            wandb.watch(self.model, log="all")
            # dummy = Variable(torch.randn(3, 512, 512))
            # dummy1 = Variable(torch.randn(args().batch_size,3, 512, 512))
            # x = ({'image':dummy1},[dummy]*args().batch_size)
            # torch.onnx.export(self.model.module, x,'model.onnx')
        
        logging.info('start training')
        self.model.train()
        if self.fix_backbone_training_scratch:
            fix_backbone(self.model, exclude_key=['backbone.'])
        else:
            train_entire_model(self.model)
            
        print(self.epoch)
        for epoch in range(self.epoch):
            print(epoch,'epoch')
            # if epoch==1:
            #     train_entire_model(self.model)
            self.train_epoch(epoch)
        self.summary_writer.close()
        
        # if args().use_wandb and args().local_rank==0:
        #     self.model.to_onnx()
        #     wandb.save("model.onnx")


    def train_step(self, meta_data, window_meta_data):
        self.optimizer.zero_grad()
        outputs = self.network_forward(self.model, meta_data, window_meta_data, self.train_cfg)
        
        if not self.model_return_loss:
            outputs.update(self._calc_loss(outputs))
        loss, outputs = self.mutli_task_uncertainty_weighted_loss(outputs)
        
    
        if self.model_precision=='fp16':
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return outputs, loss

    def train_log_visualization(self, outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index):
        losses.update(loss.item())
        losses_dict.update(outputs['loss_dict'])
        
        if self.global_count%self.print_freq==0:
            message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(
                      epoch, iter_index + 1,  len(self.loader), losses_dict.avg(), #Acc {3} | accuracies.avg(), 
                      data_time=data_time, run_time=run_time, loss=losses, lr = self.optimizer.param_groups[0]['lr'])
            # print(message)
            write2log(self.log_file,'%s\n' % message)
            self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
            self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)
            
            losses.reset(); losses_dict.reset(); data_time.reset() #accuracies.reset(); 
            self.summary_writer.flush()

        if self.global_count%(6*self.print_freq)==0 or self.global_count==50:
            vis_ids, vis_errors = determ_worst_best(outputs['kp_error'], top_n=3)
            save_name = '{}'.format(self.global_count)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            train_vis_dict = self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=['org_img', 'mesh', 'pj2d', 'centermap'],\
                vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.train_img_dir, 'save_name':save_name, 'verrors': [vis_errors], 'error_names':['E']})
    
    
    # TO DO: Correct naming here
    def get_window(self, img_info, **kwargs):
        """
        Loads the frames around a specific index.
        """            
        meta_data = list()
        for i,path in enumerate(img_info['imgpath']):
            img_path = img_info['imgpath'][i]
            video_name = img_info['video_name'][i] 
            dataset = dataset_dict[img_info['data_class'][i]](**kwargs)       
            path = img_path.split('/')
            root = "/".join(path[:-1])
            img_num = path[-1].split("_")[1]
            frame_index = int(img_num.split(".")[0])

            if frame_index-10>=0:
                prev_frame = random.choice(list(range(frame_index-10,frame_index)))
            elif frame_index>0:
                prev_frame = random.choice(list(range(0,frame_index)))
            elif frame_index==0:
                prev_frame = random.choice(list(range(0,frame_index+1)))
                 
            prev_frame_path = os.path.join(root,"image_"+"0"*(5-len(str(prev_frame)))+str(prev_frame)+".jpg")
            prev_image = dataset.get_image_from_video_name(prev_frame_path)
            meta_data.append(prev_image) 
                                                         
        return meta_data
    
    def train_epoch(self,epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict= AverageMeter_Dict()
        batch_start_time = time.time()
        
        # print(self.model_save_dir,333)
        for iter_index, meta_data in enumerate(self.loader): 
            window_meta_data = self.get_window(meta_data)
            window_meta_data = torch.stack(window_meta_data,axis=0)
            
            #torch.cuda.reset_peak_memory_stats(device=0)
            self.global_count += 1
            if args().new_training:
                if self.global_count==args().new_training_iters:
                    self.train_cfg['new_training'],self.val_cfg['new_training'],self.eval_cfg['new_training'] = False, False, False

            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()

            outputs, loss = self.train_step(meta_data,window_meta_data)
            
            if args().use_wandb and args().local_rank==0:
                wandb.log({'epoch': epoch, 'loss': loss.item(),'metrics':outputs})
            
            if self.local_rank in [-1, 0]:
                run_time.update(time.time() - run_start_time)
                self.train_log_visualization(outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index)
            
            if self.global_count%self.test_interval==0 or self.global_count==self.fast_eval_iter: #self.print_freq*2
                save_model(self.model,'{}_val_cache.pkl'.format(self.tab),parent_folder=self.model_save_dir)
                self.validation(epoch)
            
            if self.distributed_training:
                # wait for rank 0 process finish the job
                torch.distributed.barrier()
            batch_start_time = time.time()
            
        title  = '{}_epoch_{}.pkl'.format(self.tab,epoch)
        save_model(self.model,title,parent_folder=self.model_save_dir)
        self.e_sche.step()

    def validation(self,epoch):
        logging.info('evaluation result on {} iters: '.format(epoch))
        for ds_name, val_loader in self.dataset_val_list.items():
            logging.info('Evaluation on {}'.format(ds_name))
            print('here')
            MPJPE, PA_MPJPE, eval_results = val_result(self,loader_val=val_loader, evaluation=False)
            if ds_name=='pw3d' and PA_MPJPE<self.val_best_PAMPJPE['pw3d']:
                self.val_best_PAMPJPE['pw3d'] = PA_MPJPE
                _, _, eval_results = val_result(self,loader_val=self.dataset_test_list['pw3d'], evaluation=True)
                self.summary_writer.add_scalars('pw3d-vibe-test', eval_results, self.global_count)
            if ds_name=='mpiinf':
                _, _, eval_results = val_result(self,loader_val=self.dataset_test_list['mpiinf'], evaluation=True)
                self.summary_writer.add_scalars('mpiinf-test', eval_results, self.global_count)
   
            self.evaluation_results_dict[ds_name]['MPJPE'].append(MPJPE)
            self.evaluation_results_dict[ds_name]['PAMPJPE'].append(PA_MPJPE)

            logging.info('Running evaluation results:')
            ds_running_results = self.get_running_results(ds_name)
            print('Running MPJPE:{}|{}; Running PAMPJPE:{}|{}'.format(*ds_running_results))

        title = '{}_{:.4f}_{:.4f}_{}.pkl'.format(epoch, MPJPE, PA_MPJPE, self.tab)
        logging.info('Model saved as {}'.format(title))
        save_model(self.model,title,parent_folder=self.model_save_dir)

        self.model.train()
        self.summary_writer.flush()

    def get_running_results(self, ds):
        mpjpe = np.array(self.evaluation_results_dict[ds]['MPJPE'])
        pampjpe = np.array(self.evaluation_results_dict[ds]['PAMPJPE'])
        mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var = np.mean(mpjpe), np.var(mpjpe), np.mean(pampjpe), np.var(pampjpe)
        return mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Trainer()
        trainer.train()

if __name__ == '__main__':
    main()
