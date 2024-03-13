import torch
import argparse
import matplotlib.pyplot as plt
import lightning
import einops as ein
import pandas as pd
import copy
import os

from sklearn.preprocessing import StandardScaler
from model.utils.transform import JointDifference, JointRelativePosition, Compose

# All available subjects categories in the dataset
SUBJECT_CATEGORIES = {
    'expert': 'CG/Expert',
    'non-expert': 'CG/NotExpert',
    'stroke': 'GPP/Stroke',
    'parkinson': 'GPP/Parkinson',
    'backpain': 'GPP/BackPain'
}

# Available exercises
EXERCISES = range(1, 6)

# Joints of the body
JOINTS_COUNT = 25

# Features to be used
FEATURES = ['pos_x', 'pos_y', 'pos_z']

# Joints that are not usefull (hands, facial, feet)
# source: https://lisajamhoury.medium.com/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16
EXCLUDED_JOINTS = [
    15, # foot_left 
    19, # foot_right
    21, # handtip_left
    22, # thumb_left
    23, # handtip_right
    24  # thumb_right
]

class KiMoReDataset(torch.utils.data.Dataset):
    """
        KInematic Assessment of MOvement and Clinical Scores for
        Remote Monitoring of Physical REhabilitation

        Each dataset item is a temporal skeleton evolution
        with a quality scores assigned

        Each sample is of shape (frames, joints, features)
    """

    def __init__(
        self,
        data_dir,           # Directory where to search for the data
        exercise,           # Single exercise to load
        window_size,        # Size of the frame windows
        window_delta,       # Offset of each window from the previous one
        train=True,         # If it is to be used for training
        leave_one_out=None, # Subject to leave out for validation (only if train == False)
        transform=[]        # Data transformation
    ):
        super().__init__()
        print(f"preparing dataset using window_size = {window_size} with window_delta = {window_delta}")

        if exercise not in EXERCISES:
            raise ValueError(f'Exercise {exercise} not in range {EXERCISES}')

        samples = []
        targets = []

        targets_df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))
        samples_df = pd.read_csv(os.path.join(data_dir, 'samples.csv'))

        # Filter data
        targets_df = targets_df.loc[targets_df['exercise'] == exercise]
        samples_df = samples_df.loc[samples_df['exercise'] == exercise]

        # Leave one subject out for all classes for test set
        if leave_one_out is not None:
            samples_df = samples_df.loc[samples_df['subject'].str.endswith(str(train)) != train]

        # Score for each subject
        score_map = {}
        for subject, target in targets_df.groupby('subject'):
            score_map[subject] = torch.tensor(target['TS'].to_numpy(), dtype=torch.float32)

        # Convert to samples
        for name, subject in samples_df.groupby('subject'):
            subject.set_index(['frame', 'joint'], inplace=True)
            subject = subject[FEATURES]
            subject.sort_index()

            # Use index to obtain tensor dimensionality
            frames = len(subject.index.get_level_values(0).unique())
            joints = len(subject.index.get_level_values(1).unique())

            if frames < window_size:
                print(f'WARNING: sample too small: {name}, frames: {frames}')
                continue

            complete = torch.tensor(subject.values, dtype=torch.float32)

            # FIXME: How can this be wrong?
            if complete.shape[0] != frames * joints:
                print(f'WARNING: shape doesn\'t match for {name}')
                continue

            complete = torch.reshape(complete, (frames, joints, len(FEATURES)))

            # the label was not saved in the dataset
            if name not in score_map:
                continue

            score = score_map[name]
            for beg in range(0, frames - window_size, window_delta):
                #print(f'sample: {beg}-{beg+window_size}')
                sample = transform(complete[beg:beg+window_size])
                samples.append(sample)
                targets.append(score)

        # Compact samples and targets to tensor
        self.samples = torch.stack(samples, dim=0)
        self.targets = torch.stack(targets, dim=0).squeeze()

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class KiMoReDataModule(lightning.LightningDataModule):
    """
        Dataloader for the KiMoRe dataset
    """

    def __init__(
        self, 
        batch_size,
        leave_one_out=None, 
        **dataset_args
    ):
        super().__init__()
        self.batch_size = batch_size
        self.leave_one_out = leave_one_out
        self.dataset_args = dataset_args
        self.scaler = StandardScaler()

    def standardize(self, data, learn):
        S, L, J, F = data.shape
        samples = ein.rearrange(data, 'S L J F -> (S L) (J F)')
        if learn:
            samples = self.scaler.fit_transform(samples)
        else:
            samples = self.scaler.transform(samples)
        
        samples = ein.rearrange(samples, '(S L) (J F) -> S L J F', S=S, L=L, J=J, F=F)
        return torch.tensor(samples, dtype=torch.float32)

    def setup(self, task=''):

        if self.leave_one_out is not None:
            self.training = KiMoReDataset(**self.dataset_args, train=True, leave_one_out=self.leave_one_out)
            self.training.samples = self.standardize(self.training.samples, learn=False)
            self.validation = KiMoReDataset(**self.dataset_args, train=False, leave_one_out=self.leave_one_out)
            self.validation.samples = self.standardize(self.validation.samples, learn=True)
        
        else:
            dataset = KiMoReDataset(**self.dataset_args)
            dataset.samples = self.standardize(dataset.samples, learn=True)
            self.training, self.validation = torch.utils.data.random_split(dataset, [0.8, 0.2])

        print(f'LOG: training   samples count: {len(self.training)}')
        print(f'LOG: validation samples count: {len(self.validation)}')
        
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

    @staticmethod
    def add_parser_args(parser):
        opts = parser.add_argument_group('dataset')
        opts.add_argument('--window_size', type=int, default=200)
        opts.add_argument('--window_delta', type=int, default=50)
        opts.add_argument('--batch_size', type=int, default=12)
        opts.add_argument('--dataset', type=str)


def _load_single_exercise(samples, data_descriptor, filepath):
    with open(filepath) as f:
        frame = 0
        required_tokens = 0
        for line in f.readlines():
            if len(line) >= 10:
                data_descriptor['frame'] = frame
                frame += 1

                tokens = line.split(',')[:-1]
                if required_tokens == 0:
                    required_tokens = len(tokens)

                if len(tokens) != required_tokens:
                    print(f"Error with {data_descriptor['subject']}")
                    return # Ops, something wrong in the data

                joint = 0
                for joint_id in range(JOINTS_COUNT):
                    if joint_id in EXCLUDED_JOINTS:
                        continue

                    data_descriptor['joint'] = joint
                    joint += 1

                    # Insert data unit
                    data_descriptor['pos_x'] = float(tokens[joint_id * 4 + 0])
                    data_descriptor['pos_y'] = float(tokens[joint_id * 4 + 1])
                    data_descriptor['pos_z'] = float(tokens[joint_id * 4 + 2])

                    # Merge data
                    for key, value in samples.items():
                        samples[key].append(data_descriptor[key])

def _load_evaluations(targets, data_descriptor, filepath):
    with open(filepath) as f:
        _, values = f.readline(), f.readline()
        tokens = values.split(',')
        #print(tokens)
        for exercise in range(5):
            # TODO: Maybe data is broken, check correct number of elements

            eval_descriptor = copy.deepcopy(data_descriptor)
            eval_descriptor['exercise'] = exercise
            eval_descriptor['TS'] = float(tokens[1 + 0 + exercise])
            eval_descriptor['PO'] = float(tokens[1 + 5 + exercise])
            eval_descriptor['CF'] = float(tokens[1 + 10 + exercise])

            # Merge data
            for key, value in targets.items():
                targets[key].append(eval_descriptor[key])

# =======================================================================================
# PREPROCESS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='folder where raw dataset is located')
    parser.add_argument('output', help='folder where to store dataset')
    args = parser.parse_args()

    # Resulting samples
    samples = {

        # Metadata
        'type': [],
        'subject': [],
        'exercise': [],
        'frame': [],
        'joint': [],

        # Effective data
        'pos_x': [],
        'pos_y': [],
        'pos_z': []
    }

    # Resulting target for each exercise
    targets = {

        # Metadata
        'type': [],
        'subject': [],
        'exercise': [],

        # Effective data
        'TS': [],
        'PO': [],
        'CF': []
    }

    # Dict for a single data entry
    data_descriptor = {}

    # Load all data
    for subject_type in SUBJECT_CATEGORIES.values():
        data_descriptor['type'] = subject_type

        path = os.path.join(args.input, subject_type)
        subjects_list = [f.name for f in os.scandir(path) if f.is_dir()]
        for subject in subjects_list:
            data_descriptor['subject'] = subject

            # The dataset is strange, all exercises folders have the same label file containing the evaluations
            # of all the exercises, such a waste of space. Process only the first one...
            loaded_evaluations = False

            # Process all exercises
            subject_path = os.path.join(path, subject)
            exercises_list = [f.name for f in os.scandir(subject_path) if f.is_dir()]
            for exercise in exercises_list:
                data_descriptor['exercise'] = int(exercise[-1:])

                #print(f'processing {subject_type}/{subject}/{exercise}')
                exercise_path = os.path.join(subject_path, exercise)

                # Process evaluation data
                if not loaded_evaluations:
                    exercises_eval_path = os.path.join(exercise_path, 'Label')
                    if os.path.exists(exercises_eval_path):
                        for file in os.scandir(exercises_eval_path):
                            if file.name.startswith('ClinicalAssessment') and file.name.endswith('.csv'):
                                _load_evaluations(targets, data_descriptor, file.path)
                                loaded_evaluations = True
                                break

                # Process frame data
                exercise_raw_path = os.path.join(exercise_path, 'Raw')
                if os.path.exists(exercise_raw_path):
                    for file in os.scandir(exercise_raw_path):
                        if file.name.startswith('JointPosition'):
                            _load_single_exercise(samples, data_descriptor, file.path)

    os.makedirs(args.output, exist_ok=True)

    data_df = pd.DataFrame.from_dict(samples)
    targets_df = pd.DataFrame.from_dict(targets)

    print(data_df)
    print(targets_df)

    data_df.to_csv(os.path.join(args.output, 'samples.csv'))
    targets_df.to_csv(os.path.join(args.output, 'targets.csv'))


