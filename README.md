# TensorFlow-Pointer-Networks

Tensorflow implementation of Pointer Networks, modified to use a threshold (or hardmax) pointer instead of a softmax pointer.
## What is a Pointer-Network?
In a normal sequence-to-sequence model, we train a recurrent neural network (RNN) to output a sequence of elements from an output dictionary. In Vinyals et al.'s `Ptr-Net` architecture, we train a RNN to choose an element of the input sequence to output. 
![Sequence-to-sequence vs Pointer Net](https://github.com/Chanlaw/pointer-networks/blob/master/seq2seqVSptr.png "Sequence-to-sequence vs Pointer Net")
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
### Tensorboard Logging
The code supports Tensorboard logging for (test) accuracy, (training) loss, and test loss. The default log directory is `./pointer_logs/`. To run Tensorboard:
```
tensorboard --log_dir=pointer_logs
```

## Reference
- Oriol Vinyals, Meire Fortunato, Navdeep Jaitly, "Pointer Networks" [arXiv:1506.03134](http://arxiv.org/abs/1506.03134)
