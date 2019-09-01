# siamese_modeller(model, input_shape=(256,256,3)) `<Function>`
Generates a Siamese Model based on the `model` passed to the function as a feature encoder.

## Examples

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

## Arguments
* model: Keras model to be trained.
* input_shape: the shape of the input.

## Outputs
* siamese_net: A siamese NN that takes 3 inputs ordered as (anchor, positve, negative) and outputs the difference between the distances between the anchor and the positive and the anchor and the negative example. 
        `Distance_A_P - Distance_A_N`

# metric(y_true, y_pred) `<Function>`
Returns the ratio of examples of the batch that satisfies `y_pred < 0`.

# loss(y_true,y_pred) `<Function>`
Returns the loss calculated `y_pred + alpha if y_pred + alpha > 0 and 0 otherwise` where alpha is a hyperparameter.

# siamese_DataGenerator() `<Class>`
sample datagenerator for Siamese Model. Check utils documentation for further info on the attributes of the class.


