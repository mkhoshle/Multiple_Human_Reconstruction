from pycocotools.coco import COCO
import sys, os

from dataset.image_base import *
from dataset.camera_parameters import cmu_panoptic_calibration_data_source 


class CMU_Panoptic_train(Image_base):
    def __init__(self,train_flag=True, validation=False, **kwargs):
        super(CMU_Panoptic_train,self).__init__(train_flag)
        self.data_folder = os.path.join(self.data_folder,'cmu_panoptic/')
        self.annots_folder = os.path.join(self.data_folder,'panoptic_annot')
        self.load_annots()

        self.image_folder = self.data_folder
        self.file_paths = list(self.annots.keys())

        for green_frame in green_frames:
            del self.annots[green_frame]

        if validation:
            set_name = 'validation'
        else:
            set_name = 'train'

        previous_sample_num = len(self.file_paths)

        self.kp2d_mapper = constants.joint_mapping(constants.Panoptic_17,constants.Panoptic_54)
        self.kp3d_mapper = constants.joint_mapping(constants.Panoptic_17,constants.Panoptic_54)
        
       
        logging.info('CMU Panoptic dataset {} set, total {} samples'.format(set_name, self.__len__()))

    def __len__(self):
        return len(self.file_paths)


    def load_annots(self):
        self.annots = {}
        for annots_file_name in os.listdir(self.annots_folder):
            ann_file = os.path.join(self.annots_folder, annots_file_name)
            with open(ann_file, 'rb') as f:
                img_infos = pickle.load(f)
            for img_info in img_infos:
                img_path = img_info['filename'].split('/')
                img_name = img_path[1]+'-'+img_path[-1].replace('.png', '.jpg')
                self.annots[img_name] = {}
                self.annots[img_name] = img_info


    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)
        

    def get_image_info(self, index):
        if self.homogenize_pose_space:
            index = self.homogenize_pose_sample(index)

        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = os.path.join(self.image_folder,img_name)
        while not os.path.exists(imgpath):
            img_name = self.file_paths[np.random.randint(len(self))]
            imgpath = os.path.join(self.image_folder,img_name)

        image = cv2.imread(imgpath)[:,:,::-1]
        video_name = imgpath.split("/")[-1].split("-")[0]
        name = imgpath.split("/")[-1].split("-")[1][:5]
        for dic in cmu_panoptic_calibration_data_source:
            if dic['video']==video_name and dic['name']==name:
                R, T = dic['R'], dic['t']
                camMats = dic['K']
                camDists = dic['distCoef']

        visible_person_id, kp2ds = self.determine_visible_person(self.annots[img_name]['kpts2d'], self.annots[img_name]['width'],self.annots[img_name]['height'])
        kp3ds = self.annots[img_name]['kpts3d'][visible_person_id]
        full_kp2d, kp_3ds, valid_mask_2d, valid_mask_3d = [], [], [], []
        for inds, (kp2d, kp3d) in enumerate(zip(kp2ds, kp3ds)):
            invis_kps = kp2d[:,-1]<0.1
            kp2d *= 1920./832.
            kp2d[invis_kps] = -2.
            kp2d = self.map_kps(kp2d[self.H36M_TO_LSP],maps=self.joint_mapper)
            kp2d[constants.SMPL_ALL_54['Head_top']] = -2.
            full_kp2d.append(kp2d)
            valid_mask_2d.append([True,False,True])
            invis_3dkps = kp3d[:,-1]<0.1
            kp3d = kp3d[:,:3]
            kp3d[invis_3dkps] = -2.
            kp3d = kp3d[self.J24_TO_H36M]
            kp3d[0] -= np.array([0.0,0.06,0.0]) #fix the skeleton misalign
            kp_3ds.append(kp3d)
            valid_mask_3d.append([True,True,True,True])

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': full_kp2d, 'track_ids': None,\
                'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d),\
                'kp3ds': kp_3ds, 'params': None, 'camMats': camMats, 'camDists': camDists, 'img_size': image.shape[:2], 'ds': 'cmup',\
                'data_class':'cmup'}
        return img_info



green_frames = ['160422_haggling1-00_16_00002945.jpg',
                '160422_haggling1-00_16_00002946.jpg',
                '160422_haggling1-00_16_00002947.jpg',
                '160422_haggling1-00_16_00002948.jpg',
                '160422_haggling1-00_16_00002949.jpg',
                '160422_haggling1-00_16_00002950.jpg',
                '160422_haggling1-00_16_00002951.jpg',
                '160422_haggling1-00_16_00002952.jpg',
                '160422_haggling1-00_16_00002953.jpg',
                '160422_haggling1-00_16_00002954.jpg',
                '160422_haggling1-00_30_00001402.jpg',
                '160422_haggling1-00_30_00001403.jpg',
                '160422_haggling1-00_30_00001404.jpg',
                '160422_haggling1-00_30_00001405.jpg',
                '160422_haggling1-00_30_00001406.jpg',
                '160422_haggling1-00_30_00001407.jpg',
                '160422_haggling1-00_30_00001408.jpg',
                '160422_haggling1-00_30_00001409.jpg',
                '160422_haggling1-00_30_00001410.jpg',
                '160422_haggling1-00_30_00001411.jpg',
                '160422_haggling1-00_30_00001412.jpg',
                '160422_haggling1-00_30_00001414.jpg']

if __name__ == '__main__':
    args().use_fit_smpl_params = False
    dataset=CMU_Panoptic_train(train_flag=True,regress_smpl=True)
    test_dataset(dataset,with_smpl=True)
    print('Done')