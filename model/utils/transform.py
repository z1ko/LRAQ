import torch

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class JointDifference():
    """Insert single step frame difference in the features
    """

    def __call__(self, sample):
        L, J, F = sample.shape
        result = torch.zeros((L-1, J, F * 2))
        result[..., :F] = sample[:-1, :, :]
        for frame in range(L-1):
            result[frame, :, F:] = sample[frame+1, :, :] - sample[frame, :, :]
        return result


class JointRemoveFeatures():
    """Remove position features
    """

    def __init__(self, to):
        self.to = to

    def __call__(self, sample):
        L, J, F = sample.shape
        return sample[..., self.to+1:F]


class JointRelativePosition():
    """Make all joint features relative to the root joints
    """

    def __init__(self, skeleton_roots_joints):
        self.root_joints = skeleton_roots_joints

    def __call__(self, sample):
        joint_root_mean = sample[:, self.root_joints, :].mean(axis=1, keepdims=True)
        return sample - joint_root_mean