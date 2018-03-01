import pathlib


def get_save_path(model, directory='./model_checkpoints', fold=None):
    model_name = model.__class__.__name__
    path = directory + '/{}/{}'.format(model_name, model_name)
    # create dirs if they don't exist.
    pathlib.Path(directory + '/{}/'.format(model_name)).mkdir(parents=True, exist_ok=True)

    if fold is not None:
        path = path + '-fold-{}'.format(fold)

    path = path + '.hdf5'

    return path


def get_submission_path(model, directory='./'):
    model_name = model.__class__.__name__
    path = directory + '{}_submission.csv'.format(model_name)
    # create dirs if they don't exist.
    pathlib.Path(directory + '{}_submission.csv'.format(model_name)).mkdir(parents=True, exist_ok=True)

    return path
