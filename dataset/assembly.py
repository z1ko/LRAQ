import torch
import os
import pickle
import lightning
import json
import pandas as pd
import numpy as np
import operator
import argparse
import tqdm


class AssemblyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        transform=[]
    ):
        super().__init__()

        self.samples = []
        self.labels = []

        # Load all data
        for file in os.listdir(os.path.join(data_dir, split)):
            with open(file, "rb") as f:
                sample = pickle.load(file)

            self.samples.append(sample['data'])
            self.labels.append(sample['label'])
            
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.samples)


class AssemblyDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, task=''):
        self.training = AssemblyDataset(self.data_dir, 'train')
        self.validation = AssemblyDataset(self.data_dir, 'validation')

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

def main(args):

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
    args.add_argument('--poses', type=str, default='../../dataset/Assembly101/poses@60fps')
    args.add_argument('--annotations', type=str, default='../../dataset/Assembly101/fine-grained-annotations')
    args.add_argument('--output', type=str, default='data/processed/assembly')
    args = args.parse_args()
    main(args)