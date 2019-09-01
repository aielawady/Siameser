# tripler(X,y,classnames=None,triplets_dist_anch={}, triplets_dist_pos = {}, triplets_dist_neg = {}) `<Function>`
Generates a list of triplets to train the Siamese Model of the following form `[anchor, positive, negative]`. The distribution of the `anchor`, `positive` and `negative` depends on the arguments passed to the function. 

## Example

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

## Arguments
* X: numpy array with the examples names.
* y: labels for the examples.
* classnames: list with labels. The default is `None` the classnames is `set(y)`.
* triplets_dist_anch: Dictionary discribes the distribution for sampling the examples of the anchor column. It should be in the following form.
```python
triplets_dist_anch = {
    'label_1': ratio_1,
    'label_2': ratio_2,
    ...
    'label_N': ratio_N,
}

```
The sum of ratios must be less than or equal to 1. The dictionary doesn't have to include all labels. The examples that aren't specified in the dictionary will be sampled from with equal probabilities for each example and sum equal to `1-sum(ratios)`.
* triplets_dist_pos: Dictionary with labels as keys and dictionary describing the distribution for positive column as values. It should be in the following form.
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
* triplets_dist_neg: Dictionary with labels as keys and dictionary describing the distribution for positive column as values. It should have the same form as triplets_dist_pos. The remaining `1 - ratio` will be picked randomly from `X[y!=label]`.

## Outputs:
* X_triplet: list of length `len(X)` each element is a list `[anchor, positive, negative]`.

# tripler_valid(X,y,classnames=None) `<Function>`
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

## Arguments
* X: numpy array with the examples names.
* y: labels for the examples.
* classnames: list with labels. The default is `None` the classnames is `set(y)`.

## Outputs
* X_triplet: list of length `2*(len(classnames)-1)*len(X)` has the distribution as describes above.

# triplets_checker(X,y, triplets, pos_method='equal') `<Function>`
Checks if the triplets follows is formed correctly based on `pos_method`.
## Example
```python
if triplets_checker(X, y, triplets):
    print('All examples have the positive as the same class of the anchor and the negative is not of the same class.')
else:
    print('The triplets aren't correct.')
```
## Arguments
* X: numpy array with the examples names.
* y: labels for the examples.
* triplet: list with each element is a list of 3 elements to be cheched.
* pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

## Outputs:
`bool` value True or False.

# triplets_dist_display(X,y,triplets) `<Function>`
prints histogram of the anchor column and 2 matrices, one for positve column and the other for the negative. Each matrix has the rows as the histogram for each class.

## Example
```python
triplets_dist_display(X, y, triplets):
```
## Arguments
* X: numpy array with the examples names.
* y: labels for the examples.
* triplet: list with each element is a list of 3 elements to be cheched.
* pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

## Outputs:
The function doesn't return anything.
