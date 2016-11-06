# Pointer-networks

Tensorflow implementation of Pointer Networks, modified to use a threshold (or hardmax) pointer instead of a softmax pointer.
## What is a Pointer Network?
In a normal sequence-to-sequence model, we train a recurrent neural network (RNN) to output a sequence of elements from an output dictionary. In Vinyals et al.'s `Ptr-Net` architecture, we train a RNN to choose an element of the input sequence to output. 

To illustrate the differences between the two models, we can consider how the two networks will treat a toy problem - that of computing the planar convex hull of the five points (generated randomly using numpy) `(0.248, 0.683), (0.986, 0.224), (0.006, 1.000), (0.127, 0.157), (0.165, 0.274)`. In this case, the input is the sequence `[(0.248, 0.683), (0.986, 0.224), (0.006, 1.000), (0.127, 0.157), (0.165, 0.274)]` and the desired output is a permutation of `1,2,3` - correspoding to the three points on the convex hull`(0.006, 1.000), (0.127, 0.157), (0.986, 0.224)`:

![Convex Hull of Five Points](https://github.com/Chanlaw/pointer-networks/blob/master/convex_hull.png "Convex Hull of Five Points")

In a sequence-to-sequence model, the decoding RNN is simply a normal RNN:
![Sequence-to-sequence Model](https://github.com/Chanlaw/pointer-networks/blob/master/seq2seq.png "Sequence-to-sequence model")
In a Pointer Network, the output of the decoding RNN is used to modulate an attention mechanism over elements of the original sequence:
![Pointer Network](https://github.com/Chanlaw/pointer-networks/blob/master/ptr-net.png "Pointer Network")

Here we introduce two new variants of the original pointer net: `Hard-Ptr-Net` and `Multi-Ptr-Net`. The difference between the three networks is what input gets fed into the pointer network during inference. In the original implementation, we take the softmax over the outputs of the pointer network and use this to blend elements of the input sequence to feed to the network.

For `Hard-Ptr-Net`, we take the maximum of the outputs and use this to choose an element of the input sequence to feed to the network.

For `Multi-Ptr-Net`, we take average of the elements of the input sequence that correspond to outputs that are greater than a threshold (`0.3` by default). (This means that the network "points" to multiple elements of the input sequence.)
## Running and Evaluating Pointer Networks
The `main.py` file contains code for building and training the pointer network. To build the original `Ptr-Net` of Vinyals et al. and train it on the Convex Hull problem, run:
```
python main.py --pointer_type=softmax --problem_type=convex_hull
```
To build the `Hard-Ptr-Net` and train it on the convex hull problem:
```
python main.py --pointer_type=one_hot --problem_type=convex_hull
```
And to build the `Multi-Ptr-Net`:
```
python main.py --pointer_type=multi_hot --problem_type=convex_hull
```

### Other Parameters
In addition to the type of pointer, we can also play around with the following parameters:
```
batch_size: Batch size. Default 128.
checkpoint_dir: Directory to store checkpoints. Default 'checkpoints'.
log_dir: Directory to store tensorboard log files. Default 'pointer_logs'. 
max_len: The problem size. Default 50.
num_steps: The number of steps to train the network for. Default 100K.
rnn_size: The number of RNN units in each layer. Default 512.
num_layers: The number of layers in the network. Default 1.
problem_type: What kind of problem to train on: one of 'convex_hull' or 'sort'. Default 'convex_hull'.
steps_per_checkpoint: Print averaged train loss, test loss, and accuracy to console after this many steps. Default 100.
learning_rate: The learning rate. Default 0.001.
to_csv: Whether or not to export averaged loss and test accuracies to CSV. Default True.

```
For example, to run a `Ptr-Net` of size `128` on the problem of sorting `10` numbers, run:
```
python main.py --pointer_type=softmax --rnn_size=128 --problem_type=sort --max_len=10
```
### Tensorboard Logging
The code supports Tensorboard logging for (test) accuracy, (training) loss, and test loss. The default log directory is `./pointer_logs/`. To run Tensorboard:
```
tensorboard --log_dir=pointer_logs
```
Then navigate to the address Tensorboard is running at. (The default should be `0.0.0.0:6006`.)
## Reference
- Oriol Vinyals, Meire Fortunato, Navdeep Jaitly, "Pointer Networks" [arXiv:1506.03134](http://arxiv.org/abs/1506.03134)
