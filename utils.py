def preprocess_data(_data):
    nul, shape = _data.shape[0], _data.shape[1:]
    print(nul)
    print(shape)
    _data = _data.reshape(nul, shape[0], shape[1], 1)
    _data = _data.astype('float32')
    _data /= 255
    return _data