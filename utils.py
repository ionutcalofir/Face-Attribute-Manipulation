import cv2
import random
import numpy as np

from dataset import Dataset

class Utils:
  def __init__(self):
    self.ds = Dataset()
    self._y_target = None

  def next_batch_train(self, n_batch=None):
    x_batch = []
    y_batch = []

    x_train, y_train, batch_end = self.ds.get_train_batch(n_batch)
    y_target = self.y_target(len(y_train[0]), len(y_train))
    for x in x_train:
      x_img = cv2.imread(x)
      x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
      x_img = self._crop_and_resize_img_celeba(x_img, 178)
      x_img = self.normalize_img(x_img)

      x_batch.append(x_img)

    y_batch = np.reshape(y_train, (-1, 1, 1, len(y_train[0])))
    y_target_batch = np.reshape(y_target, (-1, 1, 1, len(y_target[0])))

    x_batch = np.array(x_batch)

    return x_batch.astype(np.float32), \
           y_batch.astype(np.float32), \
           y_target_batch.astype(np.float32), \
           batch_end

  def next_batch_val(self, n_batch=None):
    x_batch = []
    y_batch = []

    x_val, y_val, batch_end = self.ds.get_val_batch(n_batch)
    y_target = self.y_target(len(y_val[0]), len(y_val))
    for x in x_val:
      x_img = cv2.imread(x)
      x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
      x_img = self._crop_and_resize_img_celeba(x_img, 178)
      x_img = self.normalize_img(x_img)

      x_batch.append(x_img)

    y_batch = np.reshape(y_val, (-1, 1, 1, len(y_val[0])))
    y_target_batch = np.reshape(y_target, (-1, 1, 1, len(y_target[0])))

    x_batch = np.array(x_batch)

    return x_batch.astype(np.float32), \
           y_batch.astype(np.float32), \
           y_target_batch.astype(np.float32), \
           batch_end

  def next_batch_test(self, n_batch=None):
    x_batch = []
    y_batch = []

    x_test, y_test, batch_end = self.ds.get_test_batch(n_batch)
    y_target = self.y_target(len(y_test[0]), len(y_test))
    for x in x_test:
      x_img = cv2.imread(x)
      x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
      x_img = self._crop_and_resize_img_celeba(x_img, 178)
      x_img = self.normalize_img(x_img)

      x_batch.append(x_img)

    y_batch = np.reshape(y_test, (-1, 1, 1, len(y_test[0])))
    y_target_batch = np.reshape(y_target, (-1, 1, 1, len(y_target[0])))

    x_batch = np.array(x_batch)

    return x_batch.astype(np.float32), \
           y_batch.astype(np.float32), \
           y_target_batch.astype(np.float32), \
           batch_end

  def y_target(self, n_labels, n_batch):
    if self._y_target is None:
      nrs = [i for i in list(range(0, 2 ** n_labels))]
      random.shuffle(nrs)

      self._y_target = []
      for nr in nrs:
        label = [int(c) for c in bin(nr)[2:]]

        diff = n_labels - len(label)
        for z in list(range(diff)):
          label.insert(0, 0)

        self._y_target.append(label)

    n_random = [random.randint(0, len(self._y_target) - 1)
                for i in list(range(0, n_batch))]
    y_batch = [self._y_target[i] for i in n_random]

    return y_batch

  def _crop_and_resize_img_celeba(self, img, height):
    height_diff = int((img.shape[0] - height) / 2)
    img = img[height_diff:-height_diff, :, :]
    img = cv2.resize(img, (64, 64))

    return img

  def normalize_img(self, img):
    img = img.astype(np.float32)
    img = img / 127.5 - 1

    return img

  def denormalize_img(self, img):
    img = (img + 1) * 127.5
    img[img > 255.] = 255.
    img[img < 0.] = 0.
    img = img.astype(np.uint8)

    return img

  def show_img(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def save_img(self, img, name):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)
