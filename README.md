# Siameser 
Siameser is a module to facilitate training a feature extractor neural network using triplet loss, ```distance between anchor and positive < distancec between anchor and negative```.

## Getting Started

1. clone the repo 

```git clone https://github.com/aielawady/Siameser.git```

2. Import the libraries

```
import Siameser.core as core
import Siameser.utils as utils
```

3. Create the triplets. 

```
train_siamese = utils.tripler(np.arange(len(train_x)), train_y, classnames=set(train_y))
```
> <b> Please note that ```utils.tripler``` currently takes the ID of the training examples, e.g. file name or index, not the data itself. </b>

4. Load the data using the IDs of the ```train_siamese```.

```
# m: number of examples, H,W,C: dimension of the examples

X_loaded = [np.zeros((m,H,W,C), dtype='float32') for i in range(3)]
for i, ID in enumerate(train_siamese.T):
    for j in range(3):
        X_loaded[j][i,:,:,:] = train_x[ID[j]]
        
```

5. Buld the Siamese Model, compile and train.

```
siamese_model = core.siamese_modeller(feature_extractor,input_shape=(H,W,C))
siamese_model.compile(...)
siamese_model.fit(X_loaded, np.zeros((len(X_loaded[0]),)), ...)
```
> <b> Please note that there's no need for the labels as the dataset is formed in this order ```anchor, positive, negative``` </b>


## Example
[MNIST Example](../master/Siameser_example_MNIST.ipynb)

<b> Check [core.md](../master/core.md) and [utils.md](../master/utils.md) for more details. <b>
