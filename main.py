import time
import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model


# Utility code.
from src.callbacks import RocAucEvaluation
from src.load_data import load_data
from src.load_glove_embeddings import load_embedding_matrix
from src.bidirectional_GRU_attention import CUDNNBiRNNAttention

TRAIN = True
PRODUCTION = True
WRITE_RESULTS = True
MAX_FEATS = 5000

# Paths to data sets
train_path = './data/train.csv'
test_path = './data/test.csv'
# Paths to glove embeddings.
glove_path = './data/embeddings/glove.6B.100d.txt'
glove_embed_dims = 100


(x_train, y_train), (x_val, y_val), word_index, num_classes = load_data(path=train_path,
                                                           max_features=MAX_FEATS)

embedding_matrix = load_embedding_matrix(glove_path=glove_path,
                                         word_index=word_index,
                                         embedding_dimensions=glove_embed_dims)

vocab_size = len(word_index) + 1

model_instance = CUDNNBiRNNAttention(num_classes=num_classes)

print(num_classes)

if TRAIN:
    print(x_train.shape)
    model = model_instance.create_model(vocab_size,
                                        embedding_matrix,
                                        input_length=x_train.shape[1],
                                        embed_dim=glove_embed_dims)

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=model_instance.BATCH_SIZE)

    checkpoint = ModelCheckpoint(model_instance.checkpoint_path, save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=6,
                               verbose=1,
                               min_delta=0.00001)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  epsilon=0.0001,
                                  mode='min', min_lr=0.0001)

    roc_auc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_val, y_val),
              epochs=model_instance.EPOCHS,
              batch_size=model_instance.BATCH_SIZE,
              callbacks=[tensorboard, checkpoint, early_stop, roc_auc])

elif WRITE_RESULTS:
    # test = pd.read_csv('./data/test.csv')
    # submission = pd.read_csv('./data/sample_submission.csv')
    # X_test = test["comment_text"].fillna("fillna").values
    model = load_model(model_instance.checkpoint_path)
