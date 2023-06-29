import torch
import numpy as np
from .h36m import H36M
from .cmu_panoptic_eval import CMU_Panoptic_eval
from .mpii import MPII
from .AICH import AICH
from .up import UP
from .pw3d import PW3D
from .internet import Internet
from .coco14 import COCO14
from .lsp import LSP
from .posetrack import Posetrack
from .crowdpose import Crowdpose
from .crowdhuman import CrowdHuman
from .mpi_inf_3dhp import MPI_INF_3DHP
from .mpi_inf_3dhp_test import MPI_INF_3DHP_TEST
from .mpi_inf_3dhp_validation import MPI_INF_3DHP_VALIDATION
from .MuCo import MuCo
from .MuPoTS import MuPoTS

import sys, os
from prettytable import PrettyTable

from dataset.image_base import *
import config
from config import args
from collections import OrderedDict

dataset_dict = {'h36m': H36M, 'mpii': MPII, 'coco': COCO14, 'posetrack':Posetrack, 'aich':AICH, 'pw3d':PW3D, 'up':UP, \
                'crowdpose':Crowdpose, 'crowdhuman':CrowdHuman, 'lsp':LSP, 'mpiinf':MPI_INF_3DHP,'mpiinf_val':MPI_INF_3DHP_VALIDATION,\
                'mpiinf_test':MPI_INF_3DHP_TEST, 'muco':MuCo, 'mupots':MuPoTS, \
                'cmup':CMU_Panoptic_eval,'internet':Internet}

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, root_datapath):
        self._path = root_datapath
        self._data = sorted(glob.glob(self._path+"/"+"*.jpg"))


    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive
    
    @property
    def start_frame(self):
        img_name = os.path.basename(self._data[0])
        return int(img_name.split("_")[1].split(".")[0])

    @property
    def end_frame(self):
        img_name = os.path.basename(self._data[-1])
        return int(img_name.split("_")[1].split(".")[0])


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 num_segments = 3,
                 frames_per_segment = 1, 
                 **kwargs):
        
        super(VideoFrameDataset, self).__init__()
        datasets_used = args().dataset.split(',')
        self.datasets = [dataset_dict[ds](**kwargs) for ds in datasets_used] 
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        self.lengths, self.partition, self.ID_num_list, self.ID_num = [], [], [], 0
        sample_prob_dict = args().sample_prob_dict
        if not 1.0001>sum(sample_prob_dict.values())>0.999:
            print('CAUTION: The sum of sampling rates is supposed to be 1, while currently we have {}, \n please properly set the sample_prob_dict {} in config.yml'\
                .format(sum(sample_prob_dict.values()), sample_prob_dict.values()))

        self.video_list = [] 
        for ds in self.datasets:
            self.video_list += [VideoRecord(seq) for seq in ds.data3d_dir]

        for ds_idx, ds_name in enumerate(datasets_used):
            self.lengths.append(len(self.datasets[ds_idx]))
            self.partition.append(sample_prob_dict[ds_name])
          
            if self.datasets[ds_idx].ID_num>0:
                self.ID_num_list.append(self.ID_num)
                self.ID_num += self.datasets[ds_idx].ID_num
            else:
                self.ID_num_list.append(0)

        dataset_info_table = PrettyTable([' ']+datasets_used)
        dataset_info_table.add_row(['Length']+self.lengths)
        dataset_info_table.add_row(['Sample Prob.']+self.partition)
        expect_length = (np.array(self.lengths)/np.array(self.partition)).astype(np.int)
        dataset_info_table.add_row(['Expected length']+expect_length.tolist())
        self.partition = np.array(self.partition).cumsum()
        dataset_info_table.add_row(['Accum. Prob.']+self.partition.astype(np.float16).tolist())
        dataset_info_table.add_row(['Accum. ID.']+self.ID_num_list)
        print(dataset_info_table)
        self.total_length = int(expect_length.max())
        logging.info('All dataset length: {}'.format(len(self)))

    def _get_ID_num_(self):
        return self.ID_num

    def __getitem__(self, idx):
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record = self.video_list[idx]

        frame_start_indices = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                      np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices
    
    def _get(self, record, frame_start_indices):
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    

    