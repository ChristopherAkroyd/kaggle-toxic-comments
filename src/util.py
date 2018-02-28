def get_save_path(model, directory='./model_checkpoints', fold=None):
    model_name = model.__class__.__name__
    path = directory + '/{}/{}'.format(model_name, model_name)

    if fold is not None:
        path = path + '-fold-{}'.format(fold)

    path = path + '.hdf5'

    return path