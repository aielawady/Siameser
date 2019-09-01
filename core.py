def siamese_modeller(model, input_shape=(256,256,3)):
    '''Generates a Siamese Model based on the `model` passed to the function as a feature encoder.
    
    # Examples
    
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

    # Arguments
        model: Keras model to be trained.
        input_shape: the shape of the input.

    # Outputs
        siamese_net: A siamese NN that takes 3 inputs ordered as (anchor, positve, negative) and outputs the difference between the distances between the anchor and the positive and the anchor and the negative example. 
        
            `Distance_A_P - Distance_A_N`

     '''
    print("Creating and training the model...")
    print("Stacking the layers...")    
    anchor_input = Input(input_shape)
    positive_input = Input(input_shape)
    negative_input = Input(input_shape)
    encoded_A = model(anchor_input)
    encoded_P = model(positive_input)
    encoded_N = model(negative_input)
    s_A_P = Subtract()([encoded_A, encoded_P])
    s_A_N = Subtract()([encoded_A, encoded_N])
    d_A_P = Multiply()([s_A_P,s_A_P])
    d_A_N = Multiply()([s_A_N,s_A_N])
    diff = Subtract()([d_A_P,d_A_N])
    laam = Lambda(lambda x: K.sum(x,axis=-1),output_shape=(1,))
    dist = laam(diff)
    siamese_net = Model(inputs=[anchor_input,positive_input,negative_input],outputs=dist)
    return siamese_net

def metric(y_true, y_pred):
    '''
    Returns the ratio of examples of the batch that satisfies `y_pred < 0`.
    '''
    return K.sum(K.cast(K.greater(K.constant(0.0,shape=(1,)),y_pred), tf.float32))/K.cast(tf.shape(y_pred)[0], tf.float32)

def loss(y_true,y_pred):
    '''
    Return the loss calculated `y_pred + alpha if y_pred + alpha > 0 and 0 otherwise` where alpha is a hyperparameter.
    '''
    return K.maximum(y_pred + alpha,0)

class siamese_DataGenerator(keras.utils.Sequence):
    '''
        sample datagenerator for Siamese Model. Check utils documentation for further info on the attributes of the class.
    '''
    'Generates data for Keras'
    def __init__(self, X,y, batch_size=32, dim=(128,128), n_channels=3,
                 n_classes=10,is_train=True, shuffle=True, is_augment=True, augmentor=None, execlude=None, n_epochs=1,
                 triplets_dist_anch = {}, triplets_dist_pos = {},triplets_dist_neg = {}):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.X = X
        self.y = y
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.is_augment = is_augment
        self.is_train = is_train
        self.execlude = execlude
        self.n_epochs = n_epochs
        self.triplets_dist_anch = triplets_dist_anch
        self.triplets_dist_neg = triplets_dist_neg
        self.triplets_dist_pos = triplets_dist_pos
        self.epoch = -1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0,:]) / self.batch_size)/self.n_epochs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
#         print(in)
        list_IDs_temp = [self.list_IDs[:, k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        self.epoch = self.epoch+1
        if self.is_train and ((self.epoch % self.n_epochs) == 0):
            print('Updating the training triplets...')
            self.list_IDs = tripler(self.X,self.y, execlude=self.execlude, triplets_dist_anch= self.triplets_dist_anch, triplets_dist_pos=self.triplets_dist_pos, triplets_dist_neg=self.triplets_dist_neg)
            self.indexes = np.arange(len(self.list_IDs[0,:]))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        elif self.is_train == False:
            self.list_IDs = tripler_valid(self.X,self.y)
            self.indexes = np.arange(len(self.list_IDs[0,:]))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [np.zeros((self.batch_size, *self.dim, self.n_channels), dtype='float16') for i in range(3)]
        y = np.empty((self.batch_size, 4096*3))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for j in range(3):
                img = load_img(src_path+name+'.jpg',  self.dim[:2])
                img = img_to_array(img)
                if(self.is_augment and self.augmentor):
                    img = self.augmentor(img)
                X[j][i,:,:,:] = img
        return X, y
