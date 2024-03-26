import torch
import os
import pickle
import lightning
import json
import pandas as pd
import einops as ein
import numpy as np
import operator
import argparse
import lmdb

from tqdm import tqdm

ACTIONS_HEADERS = ['action_id', 'verb_id', 'noun_id', 'action_cls', 'verb_cls', 'noun_cls']

def load_action_dictionary(filepath):
    actions = pd.read_csv(filepath, header=0, names=ACTIONS_HEADERS)
    actions_dict, label_dict = dict(), dict()
    for _, act in actions.iterrows():
        actions_dict[act['action_cls']] = int(act['action_id'])
        label_dict[int(act['action_id'])] = act['action_cls']

    return actions_dict, label_dict

class AssemblyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        window_size,
        transform=[]
    ):
        super().__init__()

        self.samples = []
        self.labels = []

        # Load all data
        base_dir = os.path.join(data_dir, split)
        for file in os.listdir(base_dir):
            with open(os.path.join(base_dir, file), "rb") as f:
                sample = pickle.load(f)

            data = sample['data']
            labels = sample['labels']

            if data.shape[1] < window_size:
                print(f'warning: sample {file} too small ({data.shape[1]}<{window_size})')
                continue

            # Merge the hands features, they can interact spatially
            data = ein.rearrange(data, 'H T J D -> T (H J) D')

            # Sliding windows
            for beg in range(0, data.shape[0] - window_size, window_size):
                self.samples.append(torch.tensor(data[beg:beg+window_size, ...], dtype=torch.float32))
                self.labels.append(torch.tensor(labels[beg:beg+window_size, :], dtype=torch.int64))
            
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.samples)


class AssemblyDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir, batch_size, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(self, task=''):
        self.training = AssemblyDataset(self.data_dir, 'train', **self.kwargs)
        self.validation = AssemblyDataset(self.data_dir, 'validation', **self.kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training,
            self.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation,
            self.batch_size,
        )

VIEWS = [
    'C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
    'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
    'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit','HMC_21179183_mono10bit', 'HMC_84358933_mono10bit'
]

class Assembly101TSMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode,
        path_to_data,
        views=VIEWS,
        max_frames=1000000
    ):
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.views = views
        self.max_frames=max_frames

        # Load all view databases
        self.path_to_data = path_to_data
        self.envs = {view: lmdb.open(f'{self.path_to_data}/TSM/{view}', 
                                     readonly=True, 
                                     lock=False) for view in self.views}
        
        # Load actions and data
        actions_path = os.path.join(self.path_to_data, 'coarse-annotations', 'actions.csv')
        self.actions = pd.read_csv(actions_path)
        self.data = self.make_database()

    def make_database(self):
        annotations_path = os.path.join(self.path_to_data, 'coarse-annotations')
        data_path = os.path.join(self.path_to_data, f'{self.mode}.csv')
        data_df = pd.read_csv(data_path)

        video_data = []
        max_len, min_len = -1, 1e6
        for _, entry in tqdm(data_df.iterrows(), total=len(data_df)):
            sample = entry.to_dict()
            
            # Skip views no required
            if sample['view'] not in self.views:
                continue

            segm_filename = f"{sample['action_type']}_{sample['video_id']}.txt"
            segm_path = os.path.join(annotations_path, "coarse_labels", segm_filename)
            segm, start_frame, end_frame = self.load_segmentation(segm_path, self.actions)

            max_len = max(max_len, len(segm))
            min_len = min(min_len, len(segm))

            # Skip data with too foo frames
            #if len(segm) < self.max_frames:
            #    continue

            sample['segm'] = segm[:min(len(segm), self.max_frames)]
            sample['start_frame'] = start_frame
            sample['end_frame'] = min(end_frame, start_frame + self.max_frames)
            video_data.append(sample)
        
        print(f'elements={len(video_data)}, max_frames={max_len}, min_frames={min_len}')
        return video_data

    def load_segmentation(self, segm_path, actions):
        labels = []
        start_indices = []
        end_indices = []

        with open(segm_path, 'r') as f:
            lines = list(map(lambda s: s.split("\n"), f.readlines()))
            for line in lines:
                start, end, lbl = line[0].split("\t")[:-1]
                start_indices.append(int(start))
                end_indices.append(int(end))
                action_id = actions.loc[actions['action_cls'] == lbl, 'action_id']
                segm_len = int(end) - int(start)
                labels.append(np.full(segm_len, fill_value=action_id.item()))

        segmentation = np.concatenate(labels)
        num_frames = segmentation.shape[0]

        # start and end frame idx @30fps
        start_frame = min(start_indices)
        end_frame = max(end_indices)
        assert num_frames == (end_frame-start_frame), \
            "Length of Segmentation doesn't match with clip length."

        return segmentation, start_frame, end_frame

    def load_features(self, data_dict):
        elements = []
        view = data_dict['view']
        with self.envs[view].begin(write=False) as e:
            for i in range(data_dict['start_frame'], data_dict['end_frame']):
                key = os.path.join(data_dict['video_id'], f'{view}/{view}_{i:010d}.jpg')
                frame_data = e.get(key.strip().encode('utf-8'))
                if frame_data is None:
                    print(f"[!] No data found for key={key}.")
                    exit(2)

                frame_data = np.frombuffer(frame_data, 'float32')
                elements.append(frame_data)

        features = np.array(elements) # [T, D]
        return features

    def __getitem__(self, idx):
        sample = {}
        data_dict = self.data[idx]
        targets = data_dict['segm']

        features = self.load_features(data_dict)
        sample['features'] = torch.tensor(features[:self.max_frames])
        sample['targets'] = torch.tensor(targets).long()
        sample['video_name'] = os.path.join(data_dict['action_type'], data_dict['video_id'], data_dict['view'])
        sample['start_frame'] = data_dict['start_frame']
        sample['end_frame'] = data_dict['end_frame']
        sample['video_id'] = data_dict['video_id']

        return sample

    def __len__(self):
        return len(self.data)

#class AssemblyTSMDataset(torch.utils.data.Dataset):
#    def __init__(
#        self, 
#        mode,
#        fold_file_name,
#        actions_dict,
#        max_frames_per_video,
#        coarse_labels_folder,
#        features_path,
#        features_size,
#        num_classes,
#        statistics_path,
#        views = ASSEMBLY_VIEWS
#    ):
#        super().__init__()
#
#        assert mode in ['train', 'val'] 
#
#        self.label_name_to_id = actions_dict
#        self.coarse_labels_folder = coarse_labels_folder
#        self.validation = True if mode == 'val' else False
#        self.max_frames_per_video = max_frames_per_video
#        self.features_size = features_size
#        self.num_classes = num_classes
#        self.fold = mode
#        self.views = views
#
#        # Data storage
#        self.env = {view: lmdb.open(f'{features_path}/{view}', readonly=True, lock=False) for view in self.views}
#        with open(statistics_path, 'rb') as f:
#            self.statistics = pickle.load(f)
#
#        self.data = self.make_dataset(fold_file_name)
#
#    def read_files(self, files, fold_file_name):
#        data = []
#        for file in files:
#            with open(os.path.join(fold_file_name, file)) as f:
#                lines = f.readlines()
#                for line in lines:
#                    data.append(line.split('\t')[0])
#        return data
#
#    def make_dataset(self, fold_file_name):
#
#        # Dedice which files to use 
#        if self.fold == 'train':
#            files = [
#                'train_coarse_assembly.txt', 
#                'train_coarse_disassembly.txt'
#            ]
#        elif self.fold == 'val':
#            files = [
#                'val_coarse_assembly.txt', 
#                'val_coarse_disassembly.txt'
#            ]
#
#        data = self.read_files(files, fold_file_name)
#
#        data_arr = []
#        for video_id in data:
#            video_id = video_id.split(".txt")[0]
#            filename = os.path.join(self.coarse_labels_folder, video_id + ".txt")
#
#            recog_content, indexs = [], []
#            with open(filename, 'r') as f:
#                lines = f.readlines()
#                for line in lines:
#                    tmp = line.split('\t')
#                    start_l, end_l, label_l = int(tmp[0]), int(tmp[1]), tmp[2]
#                    indexs.extend([start_l, end_l])
#                    recog_content.extend([label_l] * (end_l - start_l))
#
#            recog_content = [self.label_name_to_id[e] for e in recog_content]
#            span = [min(indexs), max(indexs)]  # [start end)
#
#            total_frames = len(recog_content)
#            assert total_frames == (span[1] - span[0])
#
#            for view in self.views:
#                type_action = video_id.split('_')[0]
#                key_id = video_id.split(type_action)[1][1:]
#
#                # Unknown video
#                if view not in self.statistics[key_id]:
#                    continue
#
#                assert self.statistics[key_id][view][0] <= span[0]
#                span[1] = min(span[1], self.statistics[key_id][view][1])
#                if span[1] <= span[0]:
#                    # the video only involves preparation, no action before it's end.
#                    continue
#
#                start_frame_arr = []
#                end_frame_arr = []
#
#                # Extract data
#                for st in range(span[0], span[1], self.max_frames_per_video):
#                    start_frame_arr.append(st)
#                    max_end = st + (self.max_frames_per_video)
#                    end_frame = max_end if max_end < span[1] else span[1]
#                    end_frame_arr.append(end_frame)
#
#                for st_frame, end_frame in zip(start_frame_arr, end_frame_arr):
#                    element = {
#                        'type': type_action, 
#                        'view': view, 
#                        'st_frame': st_frame, 
#                        'end_frame': end_frame,
#                        'video_id': key_id, 
#                        'tot_frames': (end_frame - st_frame)
#                    }
#                    element["labels"] = np.array(recog_content[st_frame - span[0]:end_frame - span[0]], dtype=int)
#                    data_arr.append(element)
#
#        print("Number of videos logged in {} fold is {}".format(self.fold, len(data_arr)))
#        return data_arr
#
#    def __getitem__(self, idx):
#        ele_dict = self.data[idx]
#        st_frame = ele_dict['st_frame']
#        end_frame = ele_dict['end_frame']
#        view = ele_dict['view']
#        vid_type = ele_dict['type']
#
#        elements = []
#        with self.env[view].begin() as e:
#            for i in range(st_frame, end_frame):
#                key = ele_dict['video_id'] + f'/{view}/{view}_{i:010d}.jpg'
#                data = e.get(key.strip().encode('utf-8'))
#                if data is None:
#                    print('no available data.')
#                    exit(2)
#                
#                data = np.frombuffer(data, 'float32')
#                assert data.shape[0] == self.features_size
#                elements.append(data)
#
#        # Cap frame count to maximum size
#        #end_frame = min(end_frame, st_frame + self.max_frames_per_video)
#
#        elements = np.array(elements).T # [D, T]
#        sample = torch.tensor(elements, dtype=torch.float32)
#        labels = torch.tensor(ele_dict['labels']).long()
#        return sample, labels
#
#    def __len__(self):
#        return len(self.data)

# How to combine multiple samples in a sigle batch
def collate(data):
    samples, targets, metadata = [], [], []
    for data_dict in data:
        samples.append(data_dict['features'])
        targets.append(data_dict['targets'])
        metadata.append({
            'video_id': data_dict['video_id'],
            'start_frame': data_dict['start_frame'],
            'end_frame': data_dict['end_frame']
        })

    # pad to longest sequence
    samples = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    return samples, targets, metadata

class AssemblyTSMModule(lightning.LightningDataModule):
    def __init__(self, path_to_data, batch_size, views=VIEWS):
        super().__init__()
        self.batch_size = batch_size
        self.path_to_data = path_to_data
        self.views = views

    def setup(self, task=''):
        self.training = Assembly101TSMDataset(mode='train', views=self.views, path_to_data=self.path_to_data)
        self.validation = Assembly101TSMDataset(mode='val', views=self.views, path_to_data=self.path_to_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training,
            self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation,
            self.batch_size,
            pin_memory=True,
            collate_fn=collate,
        )

# =======================================================================================
# PREPROCESS

def gather_annotations(annotations, json_list_poses):
    result = dict()
    grouped = annotations.groupby(['root', 'start_frame', 'end_frame']).first().reset_index()
    for _, data in tqdm.tqdm(grouped.iterrows(), 'gathering annotations', total=len(grouped)):
        video_id_json = data.root.strip() + '.json'

        # If the hand poses for the video is not available, skip it
        if video_id_json not in json_list_poses:
            continue

        data = {
            'start_frame': data.start_frame,
            'end_frame': data.end_frame,
            'action': data.action_id,
            'noun': data.noun_id,
            'verb': data.verb_id,
            'action_cls': data.action_cls,
            'toy_id': data.toy_id,
        }

        if video_id_json not in result:
            result[video_id_json] = []
            
        result[video_id_json].append(data)

    return result

def main_poses(args):

    json_list_poses = os.listdir(args.poses)
    json_list_poses.sort()

    # CSV columns
    columns = [
        "id", "video", "start_frame", "end_frame",
        "action_id", "verb_id", "noun_id",
        "action_cls", "verb_cls", "noun_cls", "toy_id",
        "toy_name", "is_shared", "is_RGB"
    ]

    splits = ["train", "validation"]
    for split in splits:

        # Load annotations for this split
        annotations = os.path.join(args.annotations, split + ".csv")
        annotations = pd.read_csv(
            annotations, header=0, low_memory=False, names=columns)
        
        # Group multi-view annotations, select only the first
        annotations['root'] = annotations['video'].str.split('/').str[0]
        del annotations['video']

        # Gather annotations and split poses
        split_poses = gather_annotations(annotations, json_list_poses)

        # Generate output dataset with annotations for this split
        os.makedirs(os.path.join(args.output, split))
        for pose, pose_annotations in tqdm.tqdm(split_poses.items(), 'processing poses', total=len(split_poses)):

            # Sort annotations for this video
            pose_annotations = sorted(
                pose_annotations, key=operator.itemgetter('start_frame'))

            # Extract data
            with open(os.path.join(args.poses, pose)) as f:
                data = json.load(f)

            hands = []
            for hand in range(0, 2):
                hands.append(
                    np.stack([np.array(data[i]['landmarks'][str(hand)], dtype='float32') for i in range(len(data))]))
            data = np.stack(hands)

            # Create label data
            labels = np.zeros((data.shape[1], 3))  # background is zero
            for annotation in pose_annotations:
                beg, end = annotation['start_frame'], annotation['end_frame']
                # annotations@30fps, poses@60fps
                labels[beg*2:(end+1)*2, 0] = int(annotation['action']) + 1
                labels[beg*2:(end+1)*2, 1] = int(annotation['verb']) + 1
                labels[beg*2:(end+1)*2, 2] = int(annotation['noun']) + 1

            # Save result
            filepath = os.path.splitext(os.path.basename(pose))[0]
            filepath = os.path.join(args.output, split, filepath + '.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump({'data': data, 'labels': labels}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
                

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true', default=False)
    args.add_argument('--poses', type=str, default='../../dataset/Assembly101/poses@60fps')
    args.add_argument('--annotations', type=str, default='../../dataset/Assembly101/fine-grained-annotations')
    args.add_argument('--output', type=str, default='data/processed/assembly')
    args = args.parse_args()

    if not args.test:
        main_poses(args)
    else:
        actions_dict, labels_dict = load_action_dictionary('/media/z1ko/2TM2/datasets/Assembly101/coarse-annotations/actions.csv')
        num_classes = len(actions_dict)

        #dataset = AssemblyTSMDataset(
        #    mode='train',
        #    fold_file_name='/media/z1ko/2TM2/datasets/Assembly101/coarse-annotations/coarse_splits',
        #    coarse_labels_folder='/media/z1ko/2TM2/datasets/Assembly101/coarse-annotations/coarse_labels',
        #    actions_dict=actions_dict,
        #    max_frames_per_video=30000,
        #    features_size=2048,
        #    num_classes=num_classes,
        #    features_path='/media/z1ko/2TM2/datasets/Assembly101/TSM/',
        #    statistics_path='/media/z1ko/2TM2/datasets/Assembly101/assembly_statistics.pkl'
        #)

        datamodule = AssemblyTSMModule(path_to_data='/media/z1ko/2TM2/datasets/Assembly101/', batch_size=10)
        datamodule.setup()

        loader = datamodule.train_dataloader()
        samples, targets, metadata = next(iter(loader))