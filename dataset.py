class Dataset:
  def __init__(self):
    self._x_train = None
    self._y_train = None
    self._x_val = None
    self._y_val = None
    self._x_test = None
    self._y_test = None

  def get_train_batch(self, n_batch=None):
    if self._x_train is None and self._y_train is None:
      self._get_dataset()
    pass

  def get_val_batch(self, n_batch=None):
    if self._x_val is None and self._y_val is None:
      self._get_dataset()
    pass

  def get_test_batch(self, n_batch=None):
    if self._x_test is None and self._y_test is None:
      self._get_dataset()
    pass

  def _get_dataset(self):
    pass

  @staticmethod
  def build_dataset():
    pass
