# PyTorch Custom Datasets, Transforms, and DataLoader

## Firstname Lastname
TODO - Update your name in this readme

TODO - Add a badge to github actions here (see references for documentation).

## Assignment Overview
In this homework we will:
- [ ] Write a custom PyTorch [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) class using the [NASA BPS Microscopy Data](https://registry.opendata.aws/bps_microscopy/).
- [ ] Write custom PyTorch [Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset) to augment the dataset.
- [ ] Write the retrieve packaged image and label data using the [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader), a PyTorch iterator which allows for batching, shuffling, and loading the data in parallel using multiprocessing workers.

## Installing Dependencies
To make sure you download all the requirements to begin this homework assignment enter pip install -r requirements.txt into your terminal.

## Writing Custom Datasets, DataLoaders, and Transforms in PyTorch
This assignment is adapted from PyTorch's tutorial linked below:
[Writing Custom Datasets, DataLoaders, and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

### The Dataset Class
To define a custom dataset, we need to inherit the Dataset class from PyTorch. We do this by creating a new class and overriding the following methods:
- `__init__`
- `__len__`
- `__getitem__`

The `__init__` method is run once when instantiating the Dataset object and requires information such as the location of the data, transforms to apply, etc. The `__len__` method returns the number of samples in our dataset. The `__getitem__` method loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, fetches and reads the image, applies the transforms, and returns the tensor image following transformation and corresponding label in a tuple.

### Transformations
We will augment the data by writing transformatioons as callable classes instead of functions that we can later use with the PyTorch Compose class to string together sequences of transformations on the images. The transforms you will write are the following:
- NormalizeBPS: Normalizes uint16 to float32 between 0-1
- ResizeBPS: Resizes images
- VFlipBPS: Vertically flips the image
- HFlipBPS: Horizontally flips the image
- RotateBPS: Rotates the image [90, 180, 270]
- RandomCropBPS: Randomly crops the image
- ToTensor: Converts a np.array image to a PyTorch Tensor (final transformation)

Using the `torchvision.transforms.Compose(tranforms:list[Tranform])` class we can specify the list of tranforms/augmentations that an image can undergo.

Augmentations are important for deep learning because they enhance the robustness of the model since it recieves variations on examples.

### DataLoader
The `torch.utils.data.DataLoader` is an iterator that allows you to batch the data (take more than one image and label at a time), shuffle the data, and load the data in parallel using multiprocessing. An example of how to call the dataloader is below:

`dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)`

### Some Notes on PyTorch Tensors
Tensors are similar to numpy's ndarrays, with the exception that they add an additional dimension to images. For example, you may have a numpy array with the dimensions (height, width, channels), where channels correspond to RGB. The Tensor representation of the image will have an additional dimension of (batch_size, channels, height, width). This is why the `ToTransform` transformation will be the last to do a final conversion of the image from numpy array to Tensor prior to calling the DataLoader.

## Files to Work On
- `custom_dataset_dataloader_transforms/dataset/bps_dataset.py`
- `custom_dataset_dataloader_transforms/dataset/augmentation.py`

## Running Tests
There are two ways to run unit tests:
- 1. Run them directly from `tests/test_bps_dataset.py` and `tests/test_augmentation.py`
- 2. Run them with GitHub Actions CI/CD Tools.

## NOTE
- It is required that you add your name and github actions workflow badge to your readme.
- Check the logs from github actions to verify the correctness of your program.
- The initial code will not work. You will have to write the necessary code and fill in the gaps.
- Commit all changes as you develop the code in your individual private repo. Please provide descriptive commit messages and push from local to your repository. If you do not stage, commit, and push git classroom will not receive your code at all.
- Make sure your last push is before the deadline. Your last push will be considered as your final submission.
- There is no partial credit for code that does not run.
- If you need to be considered for partial grade for any reason (failing tests on github actions,etc). Then message the staff on discord before the deadline. Late requests may not be considered.

## References
[GH Badges](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)

## Authors
- @nadia-eecs Nadia Ahmed
- @campjake Jacob Campbell

