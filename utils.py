def tripler(X,y,classnames=None,triplets_dist_anch={}, triplets_dist_pos = {}, triplets_dist_neg = {}):
    '''
    Generates a list of triplets to train the Siamese Model of the following form `[anchor, positive, negative]`. The distribution of the `anchor`, `positive` and `negative` depends on the arguments passed to the function. 

    # Example

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

    # Arguments
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

    # Outputs:
        X_triplet: list of length `len(X)` each element is a list `[anchor, positive, negative]`.

     '''
    y = y.astype(int)
    if not classnames:
        classnames = set(y)
    Pr = np.zeros(y.shape)
    rem_mask = np.array([False]*len(y))
    for class_name in classnames:
        if str(class_name) in triplets_dist_anch: 
            Pr[(y==class_name)] = triplets_dist_anch[str(class_name)]/np.sum((y==class_name))
        else:
            rem_mask[y==class_name] = True
    if np.sum(rem_mask) > 0:
        Pr[rem_mask] = (1-np.sum(Pr))/np.sum(rem_mask)
    indexes_picker = np.random.choice(np.arange(len(y)), len(X), p=Pr)
    X_anch = X[indexes_picker]
    y_anch = y[indexes_picker]
    anchor = np.empty(len(X), dtype=object)
    positive = np.empty(len(X), dtype=object)
    negative = np.empty(len(X), dtype=object)
    index_arr = 0
    for class_name in classnames:
        anchor_tmp = X_anch[(y_anch==class_name)]
        if len(anchor_tmp) == 0:
            continue
        anchor[index_arr:len(anchor_tmp)+index_arr] = anchor_tmp
        index_pos = index_arr
        if str(class_name) in triplets_dist_pos: 
            tmp_dict = triplets_dist_pos[str(class_name)]
            assert(sum(tmp_dict.values()) <= 1)
            for c in tmp_dict.keys():
                ln = int(tmp_dict[c]*len(anchor_tmp))
                class_name_2 = int(c)
                positive[index_pos:ln+index_pos] = list(np.random.choice(X[(y==class_name_2)], ln))
                index_pos += ln
        remaining = len(anchor_tmp) - (index_pos - index_arr)
        assert(remaining >= 0)
        if remaining > 0:
            positive[index_pos:remaining+index_pos] = (list(np.random.choice(X[(y==class_name)], remaining)))
        index_neg = index_arr
        if str(class_name) in triplets_dist_neg: 
            tmp_dict = triplets_dist_neg[str(class_name)]
            assert(sum(tmp_dict.values()) <= 1)
            for c in tmp_dict.keys():
                ln = int(tmp_dict[c]*len(anchor_tmp))
                class_name_2 = int(c)
                negative[index_neg:ln+index_neg] = list(np.random.choice(X[(y==class_name_2)], ln))
                index_neg += ln
        
        remaining = len(anchor_tmp) - (index_neg - index_arr)
        assert(remaining >= 0)
        if remaining > 0:
            negative[index_neg:remaining+index_neg] = (list(np.random.choice(X[(y!=class_name)], remaining)))
        index_arr += len(anchor_tmp)
    X_triplet = np.append(anchor.reshape(1, -1), np.append(positive.reshape(1, -1), negative.reshape(1, -1), axis=0), axis=0)
    return X_triplet

def tripler_valid(X,y,classnames=None):
    '''
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

    # Arguments
        X: numpy array with the examples names.
        y: labels for the examples.
        classnames: list with labels. The default is `None` the classnames is `set(y)`.

    # Outputs
        X_triplet: list of length `2*(len(classnames)-1)*len(X)` has the distribution as describes above.

    '''
    if not classnames:
        classnames = set(y)
    anchor = np.empty(2*(len(classnames)-1)*len(X), dtype=object)
    positive = np.empty(2*(len(classnames)-1)*len(X), dtype=object)
    negative = np.empty(2*(len(classnames)-1)*len(X), dtype=object)
    index_arr = 0
    for class_1 in classnames:
        for class_2 in classnames:
            if class_1 == class_2:
                continue
            picked_Xs = X[(y == class_1) | (y == class_2)]
            picked_Ys = y[(y == class_1) | (y == class_2)]
            classnames_2s = set([class_1,class_2])
            for class_name in classnames_2s:
                anchor_tmp = picked_Xs[(picked_Ys==class_name)]
                anchor[index_arr:len(anchor_tmp)+index_arr] = anchor_tmp
                positive[index_arr:len(anchor_tmp)+index_arr] = (list(np.random.choice(picked_Xs[(picked_Ys==class_name)], (len(anchor_tmp)))))
                negative[index_arr:len(anchor_tmp)+index_arr] = (list(np.random.choice(picked_Xs[(picked_Ys!=class_name)], (len(anchor_tmp)))))
                index_arr += len(anchor_tmp)
    X_triplet_valid = np.append(anchor.reshape(1, -1), np.append(positive.reshape(1, -1), negative.reshape(1, -1), axis=0), axis=0)
    return X_triplet_valid

def triplets_checker(X,y, triplets, pos_method='equal'):
    '''
    Checks if the triplets follows is formed correctly based on `pos_method`.
    # Example
    ```python
    if triplets_checker(X, y, triplets):
        print('All examples have the positive as the same class of the anchor and the negative is not of the same class.')
    else:
        print('The triplets aren't correct.')
    ```

    # Arguments
        X: numpy array with the examples names.
        y: labels for the examples.
        triplet: list with each element is a list of 3 elements to be cheched.
        pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

    # Outputs:
        bool value True or False.
    '''


    if pos_method=='equal':
        return all([(y[X==i] == y[X==j]) and (y[X==i] != y[X==k]) for i,j,k in triplets.T])
    elif pos_method=='greater':
        return all([(y[X==i] >= y[X==j]) and (y[X==i] != y[X==k]) for i,j,k in triplets.T])

def triplets_dist_display(X,y,triplets):
    '''
    prints histogram of the anchor column and 2 matrices, one for positve column and the other for the negative. Each matrix has the rows as the histogram for each class.

    # Example
    ```python
    triplets_dist_display(X, y, triplets):
    ```

    # Arguments
        X: numpy array with the examples names.
        y: labels for the examples.
        triplet: list with each element is a list of 3 elements to be cheched.
        pos_method: if the value is `equal` the criteria for checking is `(anch==pos) and (anch!=neg)'. if the value is `greater` the criteria for checking is `(anch>=pos) and (anch!=neg)'.

    # Outputs:
        None.
    '''
    y_s = np.zeros((len(triplets.T),3))-1
    for ind in range(len(triplets.T)):
        i,j,k = triplets.T[ind]
        [y[X==i],y[X==j],y[X==k]] 
        y_s[ind,:] = [y[X==i],y[X==j],y[X==k]]
    pos_hist = np.zeros((5,5))
    neg_hist = np.zeros((5,5))
    anch_hist = np.histogram(y_s[:,0], bins=[0, 1, 2, 3,4,5])[0]
    for i in range(5):
        pos_hist[i,:] = np.histogram(y_s[:,1][y_s[:,0]==i], bins=[0, 1, 2, 3,4,5])[0]
    for i in range(5):
        neg_hist[i,:] = np.histogram(y_s[:,2][y_s[:,0]==i], bins=[0, 1, 2, 3,4,5])[0]
    print(anch_hist,'\n',pos_hist,'\n',neg_hist)
