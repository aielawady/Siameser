# Namespace `Siameser` {#Siameser}





    
## Sub-modules

* [Siameser.core](#Siameser.core)
* [Siameser.utils](#Siameser.utils)






    
# Module `Siameser.core` {#Siameser.core}







    
## Functions


    
### Function `loss` {#Siameser.core.loss}



    
> `def loss(alpha=0.2)`





    
### Function `metric` {#Siameser.core.metric}



    
> `def metric(y_true, y_pred)`


Returns the ratio of examples of the batch that satisfies `y_pred < 0`.


    
### Function `siamese_modeller` {#Siameser.core.siamese_modeller}



    
> `def siamese_modeller(model, input_shape=(256, 256, 3))`


Generates a Siamese Model based on the `model` passed to the function as a feature encoder.

##### Examples

```python
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    siamese_model = siamese_modeller(model, input_shape=input_shape)   
```

```python
    input_shape = (224,224,3)
    input_tensor = Input(shape=input_shape)
    base_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=None, pooling='max')
    x = Dense(1024, activation='relu')(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(128, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=[x])
    siamese_model = siamese_modeller(model, input_shape=input_shape)   
```

##### Arguments
    model: Keras model to be trained.
    input_shape: the shape of the input.

##### Outputs
    siamese_net: A siamese NN that takes 3 inputs ordered as (anchor, positve, negative) and outputs the difference between the distances between the anchor and the positive and the anchor and the negative example. 
    
        `Distance_A_P - Distance_A_N`



    
## Classes


    
### Class `siamese_DataGenerator` {#Siameser.core.siamese_DataGenerator}



> `class siamese_DataGenerator(X, y, batch_size=32, dim=(128, 128), n_channels=3, n_classes=10, is_train=True, shuffle=True, is_augment=True, augmentor=None, execlude=None, n_epochs=1, triplets_dist_anch={}, triplets_dist_pos={}, triplets_dist_neg={})`


sample datagenerator for Siamese Model. Check utils documentation for further info on the attributes of the class.

Initialization



    
#### Ancestors (in MRO)

* [keras.utils.data_utils.Sequence](#keras.utils.data_utils.Sequence)






    
#### Methods


    
##### Method `on_epoch_end` {#Siameser.core.siamese_DataGenerator.on_epoch_end}



    
> `def on_epoch_end(self)`


Updates indexes after each epoch




    
# Module `Siameser.utils` {#Siameser.utils}







    
## Functions


    
### Function `tripler` {#Siameser.utils.tripler}



    
> `def tripler(X, y, classnames=None, triplets_dist_anch={}, triplets_dist_pos={}, triplets_dist_neg={})`


Generates a list of triplets to train the Siamese Model of the following form `[anchor, positive, negative]`. The distribution of the `anchor`, `positive` and `negative` depends on the arguments passed to the function. 

##### Example

```python
triplets_dist_anch = {}
triplets_dist_pos = {
    '1':{
        '0':0.8,
    },
    '2':{
        '0':0.1,
        '1':0.7,
    },
    '3':{
        '0':0.05,
        '1':0.15,
        '2':0.6,
    }
}
triplets_dist_neg = {
    '1':{
        '2':0.7,
        '3':0.2,
    },
    '2':{
        '3':0.8,
    },
    '3':{
        '2':0.1,
        '4':0.8
    }
}

triplets = tripler(X,y, triplets_dist_anch=triplets_dist_anch, triplets_dist_pos=triplets_dist_pos, triplets_dist_neg=triplets_dist_neg)

```

##### Arguments
    X: numpy array with the examples names.
    y: labels for the examples.
    classnames: list with labels. The default is `None` the classnames is `set(y)`.
    triplets_dist_anch: Dictionary discribes the distribution for sampling the examples of the anchor column. It should be in the following form.
        ```python
        triplets_dist_anch = {
            'label_1': ratio_1,
            'label_2': ratio_2,
            ...
            'label_N': ratio_N,
        }

        ```
    The sum of ratios must be less than or equal to 1. The dictionary doesn't have to include all labels. The examples that aren't specified in the dictionary will be sampled from with equal probabilities for each example and sum equal to `1-sum(ratios)`.
    triplets_dist_pos: Dictionary with labels as keys and dictionary describing the distribution for positive column as values. It should be in the following form.
        ```python
        triplets_dist_pos = {
            'label_1': {
                'label_1': ratio_1,
                'label_2': ratio_2,
                ...
                'label_N': ratio_N,                       
            },
            'label_2': {
                'label_1': ratio_1,
                'label_2': ratio_2,
                ...
                'label_N': ratio_N,
            },
            ...
            'label_N': {
                'label_1': ratio_1,
                'label_2': ratio_2,
                ...
                'label_N': ratio_N,
            }
        }

        ```
    The sum of ratios must be less than or equal to 1. None of the dictionaries have to include all labels. The remaining `1 - ratio` will be picked randomly from `X[y==label]`.
    triplets_dist_neg: Dictionary with labels as keys and dictionary describing the distribution for positive column as values. It should have the same form as triplets_dist_pos. The remaining `1 - ratio` will be picked randomly from `X[y!=label]`.

##### Outputs:
    X_triplet: list of length `len(X)` each element is a list `[anchor, positive, negative]`.


    
### Function `tripler_valid` {#Siameser.utils.tripler_valid}



    
> `def tripler_valid(X, y, classnames=None)`


Generates triplets for validation. The elements of each class are repeated the number of other classes each instance for one class. It has the following form.
```python

['label_1', 'label_1','label_2']
['label_1', 'label_1','label_2']
...
['label_1', 'label_1','label_3']
['label_1', 'label_1','label_3']
...
['label_1', 'label_1','label_N']
['label_1', 'label_1','label_N']
['label_2', 'label_2','label_1']
['label_2', 'label_2','label_1']
...
['label_2', 'label_2','label_3']
['label_2', 'label_2','label_3']
...
['label_2', 'label_2','label_N']
['label_2', 'label_2','label_N']
...
['label_N', 'label_N','label_1']
['label_N', 'label_N','label_1']
...
['label_N', 'label_N','label_N-1']
['label_N', 'label_N','label_N-1']

``` 

##### Arguments
    X: numpy array with the examples names.
    y: labels for the examples.
    classnames: list with labels. The default is `None` the classnames is `set(y)`.

##### Outputs
    X_triplet: list of length `2*(len(classnames)-1)*len(X)` has the distribution as describes above.


    
### Function `triplets_checker` {#Siameser.utils.triplets_checker}



    
> `def triplets_checker(X, y, triplets, pos_method='equal')`


Checks if the triplets follows is formed correctly based on `pos_method`.
##### Example
```python
if triplets_checker(X, y, triplets):
    print('All examples have the positive as the same class of the anchor and the negative is not of the same class.')
else:
    print('The triplets aren't correct.')
```

##### Arguments
    X: numpy array with the examples names.
    y: labels for the examples.
    triplet: list with each element is a list of 3 elements to be cheched.
    pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

##### Outputs:
    bool value True or False.


    
### Function `triplets_dist_display` {#Siameser.utils.triplets_dist_display}



    
> `def triplets_dist_display(X, y, triplets)`


prints histogram of the anchor column and 2 matrices, one for positve column and the other for the negative. Each matrix has the rows as the histogram for each class.

##### Example
```python
triplets_dist_display(X, y, triplets):
```

##### Arguments
    X: numpy array with the examples names.
    y: labels for the examples.
    triplet: list with each element is a list of 3 elements to be cheched.
    pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

##### Outputs:
    None.

