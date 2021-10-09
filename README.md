# TransEvolve

Implementation of [Redesigning the Transformer Architecture with Insights from Multi-particle Dynamical Systems](https://arxiv.org/abs/2109.15142), accepted as spotlight paper in NeurIPS 2021.
This implementation is based on Tensorflow 2.4.1.

## Usage
After cloning the repository, 
```python
from TransEvolve.TransEvolve import models
```
#### Sequence-to-sequence learning

To use the full encoder-decoder TransEvolve model, use
```python
model = models.EncoderDecoderModel(
               input_dim=32000,
               hidden_dim=512,
               num_head=8,
               projection_type='full',
               num_layers_per_block=3,
               num_blocks=2,
               dropout=0.1)
```
where 
* `projection_type can be` 'random' (random rotation matrix approximation) or 'full' (usual full rank learnable matrices);
* `num_layers_per_block` and `num_blocks` control the degree of temporal attention propagation; for example, with  `num_layers_per_block=3` and `num_blocks=2`, the usual query-key dot
product will be computed twice and evolved three times each;
* `input_dim` denotes the size of the vocabulary for one-hot encoded sequences;

#### Encoder-only tasks

Similarly, only the TransEvolve encoder can be used as:
```python
model = models.EncoderModel(
               input_dim=32000,
               hidden_dim=512,
               num_head=8,
               projection_type='random',
               num_layers_per_block=3,
               num_blocks=2,
               dropout=0.1)
```
Encoder-only model will output a sequence of encoded representations. Alternatively, if one seeks to use TransEvolve for sequence classification tasks, one may use 
```python
model = models.ClassificationModel(
               input_dim=32000,
               hidden_dim=512,
               num_head=8,
               projection_type='random',
               num_layers_per_block=3,
               num_blocks=2,
               num_classes=4,
               dropout=0.1)
```
`ClassificationModel` uses a single output projection layer on top of TransEvolve encoder. By default, it uses mean-pooling over the encoder output sequence. For `num_classes>2` it will apply softmax over the logits, else sigmoid.


For any query, feel free to contact Subhabrata Dutta at subha0009@gmail.com.
