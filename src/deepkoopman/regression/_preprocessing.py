"""module for implementing a neural network DMD"""
from __future__ import annotations

import pickle
from abc import abstractmethod
from warnings import warn

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SeqDataDataset(Dataset):
    """
    A PyTorch Dataset class to handle sequential data in the format of (x, y, ys),
    where x is the input sequence, y is the target output sequence and ys is a vector
    indicating the maximum look-ahead distance.

    Args:
        x (torch.Tensor): The input sequence tensor of shape (batch_size,
            sequence_length, input_size).
        y (torch.Tensor): The output sequence tensor of shape (batch_size,
            sequence_length, output_size).
        ys (torch.Tensor): The maximum look-ahead distance tensor of shape
            (batch_size,).
        transform (callable, optional): Optional normalization function to apply to
            x and y.

    Returns:
        torch.Tensor: The preprocessed input sequence tensor.
        torch.Tensor: The preprocessed target output sequence tensor.
        torch.Tensor: The maximum look-ahead distance tensor.
    """

    def __init__(self, x, y, ys, transform=None):
        self.x = x.squeeze(1)
        self.y = y
        self.ys = ys
        self.normalization = transform

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx].clone()
        ys = self.ys[idx].clone()

        if self.normalization:
            x = self.normalization(x)
            y = self.normalization(y)

        return x, y, ys

class TensorNormalize(nn.Module):
    """
    Normalizes the input tensor by subtracting the mean and dividing by the standard
    deviation.

    Args:
        mean (float or tensor): The mean value to be subtracted from the input tensor.
        std (float or tensor): The standard deviation value to divide the input tensor
            by.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor):
        """
        Forward pass of the normalization module.

        Args:
            tensor (tensor): The input tensor to be normalized.

        Returns:
            The normalized tensor.
        """
        return torch.divide((tensor - self.mean), self.std)
        # return # tensor.copy_(tensor.sub_(self.mean).div_(self.std))

    def __repr__(self) -> str:
        """
        Returns a string representation of the TensorNormalize module.

        Returns:
            A string representation of the module.
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class InverseTensorNormalize(nn.Module):
    """
    A PyTorch module that performs inverse normalization on input tensors using
    a given mean and standard deviation.

    Args:
        mean (float or sequence): The mean used for normalization.
        std (float or sequence): The standard deviation used for normalization.

    Example:
        >>> mean = [0.5, 0.5, 0.5]
        >>> std = [0.5, 0.5, 0.5]
        >>> inv_norm = InverseTensorNormalize(mean, std)
        >>> normalized_tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
        >>> output = inv_norm(normalized_tensor)

    Attributes:
        mean (float or sequence): The mean used for normalization.
        std (float or sequence): The standard deviation used for normalization.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor):
        return torch.multiply(tensor, self.std) + self.mean
        # return tensor.copy_(tensor.mul_(self.std).add_(self.mean))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class SeqDataModule(L.LightningDataModule):
    """
    Class for creating sequence data dataloader for training and validation.

    Args:
        data_tr: List of 2D numpy.ndarray representing training data trajectories.
        data_val: List of 2D numpy.ndarray representing validation data trajectories.
            Can be None.
        look_forward: Number of time steps to predict forward.
        batch_size: Size of each batch of data.
        normalize: Whether to normalize the input data or not. Default is True.
        normalize_mode: The type of normalization to use. Either "equal" or "max".
            Default is "equal".
        normalize_std_factor: Scaling factor for standard deviation during
            normalization. Default is 2.0.

    Methods:
        prepare_data(): Prepares the data by converting to time-delayed data and
            computing mean and std if normalize is True.
        setup(stage=None): Sets up training and validation datasets.
        train_dataloader(): Returns a DataLoader for training data.
        val_dataloader(): Returns a DataLoader for validation data.
        convert_seq_list_to_delayed_data(data_list, look_back, look_forward): Converts
            list of sequences to time-delayed data.
        collate_fn(batch): Custom collate function to be used with DataLoader.

    Returns:
        A SeqDataModule object.
    """

    def __init__(
        self,
        data_tr,
        data_val,
        look_forward=10,
        batch_size=32,
        normalize=True,
        normalize_mode="equal",
        normalize_std_factor=2.0,
    ):
        """
        Initialize a SeqDataModule.

        Args:
            data_tr (Union[str, List[np.ndarray]]): Training data. Can be either a
                list of 2D numpy arrays, each 2D numpy array representing a trajectory,
                or the path to a pickle file containing such a list.
            data_val (Optional[Union[str, List[np.ndarray]]]): Validation data.
                Can be either a list of 2D numpy arrays, each 2D numpy array
                    representing a trajectory, or the path to a pickle file
                    containing such a list.
            look_forward (int): Number of time steps to predict into the future.
            batch_size (int): Number of samples per batch.
            normalize (bool): Whether to normalize the data. Default is True.
            normalize_mode (str): Mode for normalization. Can be either "equal"
                or "max". "equal" divides by the standard deviation, while "max"
                divides by the maximum absolute value of the data. Default is "equal".
            normalize_std_factor (float): Scaling factor for the standard deviation in
                normalization. Default is 2.0.

        Returns:
            None.
        """
        super().__init__()
        # input data_tr or data_val is a list of 2D np.ndarray. each 2d
        # np.ndarray is a trajectory, and the axis 0 is number of samples, axis 1 is
        # the number of system state
        self.data_tr = data_tr
        self.data_val = data_val
        self.look_forward = look_forward
        self.batch_size = batch_size
        self.look_back = 1
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.normalization = None
        self.inverse_transform = None
        self.normalize_std_factor = normalize_std_factor

    def prepare_data(self):
        """
        Preprocesses the input training and validation data by checking their types,
        checking for normalization, finding the mean and standard deviation of
        the training data (if normalization is enabled), and creating time-delayed data
        from the input data.

        Raises:
            ValueError: If the training data is None or has an invalid type.
            ValueError: If the validation data has an invalid type.
            TypeError: If the data is complex or not float.

        """
        # train data
        if self.data_tr is None:
            raise ValueError("You must feed training data!")
        if isinstance(self.data_tr, list):
            data_list = self.data_tr
        elif isinstance(self.data_tr, str):
            f = open(self.data_tr, "rb")
            data_list = pickle.load(f)
        else:
            raise ValueError("Wrong type of `self.data_tr`")

        # check train data
        data_list = self.check_list_of_nparray(data_list)

        # find the mean, std
        if self.normalize:
            stacked_data_list = np.vstack(data_list)
            mean = stacked_data_list.mean(axis=0)
            std = stacked_data_list.std(axis=0)

            # zero mean so easier for downstream
            self.mean = torch.FloatTensor(mean) * 0
            # default = 2.0, more stable
            self.std = torch.FloatTensor(std) * self.normalize_std_factor

            if self.normalize_mode == "max":
                self.std = torch.ones_like(self.std) * self.std.max()

            # prevent divide by zero error
            for i in range(len(self.std)):
                if self.std[i] < 1e-6:
                    self.std[i] += 1e-3

            # get transform
            self.normalization = TensorNormalize(self.mean, self.std)

            # get inverse transform
            self.inverse_transform = InverseTensorNormalize(self.mean, self.std)

        # create time-delayed data
        self._tr_x, self._tr_yseq, self._tr_ys = self.convert_seq_list_to_delayed_data(
            data_list, self.look_back, self.look_forward
        )

        # validation data
        if self.data_val is not None:
            # raise ValueError("You need to feed validation data!")
            if isinstance(self.data_val, list):
                data_list = self.data_val
            elif isinstance(self.data_val, str):
                f = open(self.data_val, "rb")
                data_list = pickle.load(f)
            else:
                raise ValueError("Wrong type of `self.data_val`")

            # check val data
            data_list = self.check_list_of_nparray(data_list)

            # create time-delayed data
            (
                self._val_x,
                self._val_yseq,
                self._val_ys,
            ) = self.convert_seq_list_to_delayed_data(
                data_list, self.look_back, self.look_forward
            )
        else:
            warn("Warning: no validation data prepared")

    def setup(self, stage=None):
        """
        Prepares the train and validation datasets for the Lightning module.
        The train dataset is created from the training data specified in the
        constructor by creating time-delayed versions of the input/output sequences.
        If `normalize` is True, the data is normalized using the mean and standard
        deviation of the training data. The validation dataset is created from the
        validation data specified in the constructor in the same way as the training
        dataset. If `normalize` is True, it is also normalized using the mean and
        standard deviation of the training data. If `stage` is not "fit",
        an exception is raised as the `setup()` method has not been implemented
        for other stages.

        Args:
            stage: The stage of training, validation or testing (default is None).

        Raises:
            NotImplementedError: If `stage` is not "fit".
        """
        # Load data and split into train and validation sets here
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.tr_dataset = SeqDataDataset(
                self._tr_x, self._tr_yseq, self._tr_ys, self.normalization
            )
            if self.data_val is not None:
                self.val_dataset = SeqDataDataset(
                    self._val_x, self._val_yseq, self._val_ys, self.normalization
                )
        else:
            raise NotImplementedError("We didn't implement for stage not `fit`")

    def train_dataloader(self):
        return DataLoader(
            self.tr_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )

    def convert_seq_list_to_delayed_data(self, data_list, look_back, look_forward):
        """
        Converts a list of sequences to time-delayed data by extracting subsequences
        of length `look_back` and `look_forward` from each sequence in the list.

        Args:
            data_list (List[np.ndarray]): A list of 2D numpy arrays. Each array
                represents a trajectory, with axis 0 representing the number of samples
                and axis 1 representing the number of system states.
            look_back (int): The number of previous time steps to include in each
                subsequence.
            look_forward (int): The number of future time steps to include in each
                subsequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three
                tensors:
            1) The time-delayed input data, with shape (num_samples, look_back,
                num_system_states).
            2) The time-delayed output data, with shape (num_samples, look_forward,
                num_system_states).
            3) The sequence lengths of the output data, with shape (num_samples,).
        """
        time_delayed_x_list = []
        time_delayed_yseq_list = []
        for seq in data_list:
            # if self.look_forward + self.look_back > len(seq):
            #     raise ValueError("look_forward too large")
            n_sub_traj = len(seq) - look_back - look_forward + 1
            if n_sub_traj >= 1:
                for i in range(len(seq) - look_back - look_forward + 1):
                    time_delayed_x_list.append(seq[i : i + look_back])
                    time_delayed_yseq_list.append(
                        seq[i + look_back : i + look_back + look_forward]
                    )
            else:
                # only 1 traj, just to predict to its end
                time_delayed_x_list.append(seq[0:1])
                time_delayed_yseq_list.append(seq[1:])
        time_delayed_yseq_lens_list = [x.shape[0] for x in time_delayed_yseq_list]

        # convert data to tensor
        time_delayed_x = torch.FloatTensor(np.array(time_delayed_x_list))
        time_delayed_yseq = pad_sequence(
            [torch.FloatTensor(x) for x in time_delayed_yseq_list], True
        )
        time_delayed_yseq_lens = torch.LongTensor(time_delayed_yseq_lens_list)
        return time_delayed_x, time_delayed_yseq, time_delayed_yseq_lens

    def collate_fn(self, batch):
        """
        Collates a batch of data.

        Args:
            batch: A list of tuples where each tuple represents a sample containing
                the input sequence `x`, the output sequence `y`, and the maximum
                number of steps to predict `ys`.

        Returns:
            A tuple containing the input sequences as a stacked tensor, the output
            sequences as a stacked tensor, and the maximum number of steps to predict
            as a stacked tensor.

        """
        x_batch, y_batch, ys_batch = zip(*batch)
        xx = torch.stack(x_batch, 0)
        yy = torch.stack(y_batch, 0)
        ys = torch.stack(ys_batch, 0)
        return xx, yy, ys

    @classmethod
    def check_list_of_nparray(cls, data_list):
        """
        Check if the input is a list of numpy arrays, and convert data to float32 if
        float64.

        Args:
            data_list (List[np.ndarray]): A list of numpy arrays representing system
                states.

        Returns:
            List[np.ndarray]: The input list of numpy arrays converted to float32.

        Raises:
            TypeError: If the input data is complex or not float.
        """
        # check if data is complex
        if any(np.iscomplexobj(x) for x in data_list):
            raise TypeError("Complex data is not supported")

        # check if data has float64
        if any(x.dtype is np.float64 for x in data_list):
            warn("Found float64 data. Will convert to float32")

        # convert data to float32 if float64
        for i, data_traj in enumerate(data_list):
            if "float" not in data_traj.dtype.name:
                raise TypeError("Found data is not float")
            if data_traj.dtype.name == "float64":
                data_list[i] = data_traj.astype("float32")

        return data_list
