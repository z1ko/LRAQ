import torch
import lightning
import einops as ein
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

___comment = """
class KiMoReDataset(torch.utils.data.Dataset):
    \"""
        KInematic Assessment of MOvement and Clinical Scores for
        Remote Monitoring of Physical REhabilitation

        Each dataset item is a temporal skeleton evolution
        with a quality scores assigned

        Each sample is of shape (frames, joints, features)
    \"""

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
    \"""
        Dataloader for the KiMoRe dataset
    \"""

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

"""

def load_kimore(filepath, exercise, fold, train, data=None):

    if data is None:
        with open(filepath, 'rb') as f:
            dataset_complete = pickle.load(f)
    else:
        dataset_complete = data

    # load only requested exercise and fold
    dataset = dataset_complete['folded'][exercise][fold]
    dataset = dataset['train' if train else 'val']

    samples = np.stack(list(map(lambda x: x['frames'], dataset)))
    targets = np.stack(list(map(lambda x: x['target'], dataset)))

    return samples, targets


class KiMoReDatasetFold(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath,
        exercise,
        fold,
        transform,
        train,
        data=None
    ):
        super().__init__()

        self.filepath = filepath
        self.exercise = exercise
        self.train = train
        self.fold = fold

        samples, targets = load_kimore(self.filepath, self.exercise, self.fold, self.train, data=data)
        self.samples = torch.from_numpy(samples).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)

        # Apply transformations
        self.samples = torch.stack([ transform(x) for x in self.samples[:] ])
        # Keep only TS assessment value
        self.targets = torch.stack([ x[..., 0].squeeze_(dim=-1) for x in self.targets[:]])
        

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class KiMoReDataModuleFolded(lightning.LightningDataModule):
    def __init__(self, filepath, batch_size, exercise, fold, transform, data=None):
        super().__init__()
        self.filepath = filepath
        self.data=data
        self.batch_size = batch_size
        self.exercise = exercise
        self.fold = fold
        self.transform = transform
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

        self.train = KiMoReDatasetFold(self.filepath, self.exercise, self.fold, self.transform, data=self.data, train=True)
        self.val = KiMoReDatasetFold(self.filepath, self.exercise, self.fold, self.transform, data=self.data, train=False)

        # Standardize data
        self.train.samples = self.standardize(self.train.samples, learn=True)
        self.val.samples = self.standardize(self.val.samples, learn=False)

        print(f'LOG: training   samples count: {len(self.train)}')
        print(f'LOG: validation samples count: {len(self.val)}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            self.batch_size,
            drop_last=False,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            self.batch_size,
        )

    @staticmethod
    def add_parser_args(parser):
        opts = parser.add_argument_group('dataset')
        opts.add_argument('--batch_size', type=int, default=12)
        opts.add_argument('--dataset', type=str)