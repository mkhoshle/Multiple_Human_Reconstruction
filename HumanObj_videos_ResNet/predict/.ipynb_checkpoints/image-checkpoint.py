import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/image.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .base_predictor import *
import constants
import glob
from utils.util import collect_image_list
from lib.dataset.internet import Internet
import matplotlib.pyplot as plt

dataset_dict = {'internet':Internet}

class Image_processor(Predictor):
    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    def get_prev_frame(self,frame_index):
        if frame_index-10 >= 0:
            prev_frame = random.choice(
                list(range(frame_index-10, frame_index)))
        elif frame_index > 0:
            prev_frame = random.choice(list(range(0, frame_index)))
        elif frame_index == 0:
            prev_frame = random.choice(list(range(0, frame_index+1)))

        return prev_frame

    def get_frame_index(self, root, data_class, img_name):
        if data_class == 'pw3d':
            img_num = img_name.split("_")[1]
            frame_index = int(img_num.split(".")[0])
            prev_frame = self.get_prev_frame(frame_index)

            prev_frame_path = os.path.join(root, "image_"+"0"*(5-len(str(prev_frame)))+str(prev_frame)+".jpg")
        elif data_class == 'mpiinf':  
            img_num = img_name.split("_")[-1]
            frame_index = int(img_num.split(".")[0][1:])
            prev_frame = frame_index-10
            
            prev_frame_path = os.path.join(root,"_".join(img_name.split("_")[:-1])+"_F"+"0"*(6-len(str(prev_frame)))+str(prev_frame)+".jpg")
            if not os.path.exists(prev_frame_path):
                prev_frame = frame_index
                prev_frame_path = os.path.join(root,"_".join(img_name.split("_")[:-1])+"_F"+"0"*(6-len(str(prev_frame)))+str(prev_frame)+".jpg")


        return frame_index, prev_frame_path

    # TO DO: Correct naming here

    def get_window(self, img_info, **kwargs):
        """
        Loads the frames around a specific index.
        """
        meta_data = list()
        for i, path in enumerate(img_info['imgpath']):
            img_path = img_info['imgpath'][i]
            dataset = dataset_dict[img_info["data_set"][i]](**kwargs)
            path = img_path.split('/')
            root = "/".join(path[:-1])
            frame_index = path[-1].split(".")[0][5:]
            prev_frame_path = root + "/image{}.jpg".format(int(frame_index)-1)
            print(frame_index,prev_frame_path,dataset)



            prev_image = dataset.get_image_from_video_name(prev_frame_path)
            if prev_image==None:
                meta_data.append(img_info['image'][i])
            else:
                meta_data.append(prev_image)

        return meta_data
    
    @torch.no_grad()
    def run(self, image_folder, tracker=None):
        print('Processing {}, saving to {}'.format(image_folder, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = self.output_dir 
        counter = Time_counter(thresh=1)

        if self.show_mesh_stand_on_image:
            from visualization.vedo_visualizer import Vedo_visualizer
            visualizer = Vedo_visualizer()
            stand_on_imgs_frames = []

        file_list = collect_image_list(image_folder=image_folder, collect_subdirs=self.collect_subdirs, img_exts=constants.img_exts)
        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
        counter.start()
        results_all = {}
        for test_iter,meta_data in enumerate(internet_loader):
            print(meta_data.keys())
            # meta_data['image'] = meta_data['image'].permute(0,3,1,2)
            # window_meta_data = self.get_window(meta_data)
            
            window_meta_data = meta_data['image']
            
            # if len(window_meta_data)>1:
            #     window_meta_data = torch.stack(window_meta_data,axis=0)
            # else:
            #     window_meta_data = window_meta_data[0]
            print(meta_data['image'].shape,window_meta_data.shape)

            # plt.imshow(meta_data['image'][0].permute(1,2, 0), cmap = "gray")
            # plt.savefig("test.png")
            
            print('cfg',self.demo_cfg)
            outputs = self.net_forward(meta_data, window_meta_data, cfg=self.demo_cfg)
            
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
                
            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                    
                # print(show_items_list)
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], \
                    show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)
                
                print(outputs.keys())
                print(outputs['centers_pred'])
                print(outputs['detection_flag'])
                print(results_dict.keys())

                # for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['mesh_rendering_orgimgs']['figs']):
                for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['centermap']['figs']):
                    save_name = os.path.join(self.output_dir, os.path.basename(img_name))
                    cv2.imwrite(save_name, cv2.cvtColor(mesh_rendering_orgimg, cv2.COLOR_RGB2BGR))

            if self.show_mesh_stand_on_image:
                stand_on_imgs = visualizer.plot_multi_meshes_batch(outputs['verts'], outputs['params']['cam'], outputs['meta_data'], \
                    outputs['reorganize_idx'].cpu().numpy(), interactive_show=self.interactive_vis)
                stand_on_imgs_frames += stand_on_imgs

            if self.save_mesh:
                save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
            
            if test_iter%8==0:
                print('Processed {} / {} images'.format(test_iter * self.val_batch_size, len(internet_loader.dataset)))
            counter.start()
            results_all.update(results)
        return results_all


def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Image_processor(args_set=args_set)
        inputs = args_set.inputs
        if not os.path.exists(inputs):
            print("Didn't find the target directory: {}. \n Running the code on the demo images".format(inputs))
            inputs = os.path.join(processor.demo_dir,'images')
        processor.run(inputs)

if __name__ == '__main__':
    main()