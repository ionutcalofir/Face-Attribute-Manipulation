import random

class Dataset:
  def __init__(self, anno_path='dataset/CelebA/anno',
                     imgs_path='dataset/CelebA/imgs'):
    self._x_train = None
    self._y_train = None
    self._x_val = None
    self._y_val = None
    self._x_test = None
    self._y_test = None
    self.labels = {'Black Hair': 0,
                   'Blond Hair': 1,
                   'Male': 2,
                   'Female': 3,
                   'Young': 4,
                   'Old': 5,
                   'Smile': 6}
    self.anno_path = anno_path
    self.imgs_path = imgs_path

    self.train_batch_idx = 0
    self.val_batch_idx = 0
    self.test_batch_idx = 0

  def get_train_batch(self, n_batch=None):
    if self._x_train is None and self._y_train is None:
      self._get_dataset()

    if n_batch is None:
      return self._x_train, self._y_train

    batch_end = False

    x_batch = [self.imgs_path + '/' + self._x_train[i]
               for i in list(range(
                 self.train_batch_idx,
                 min(self.train_batch_idx + n_batch,
                     len(self._x_train))))]
    y_batch = [self._y_train[i]
               for i in list(range(
                 self.train_batch_idx,
                 min(self.train_batch_idx + n_batch,
                     len(self._y_train))))]

    if self.train_batch_idx + n_batch < len(self._x_train):
      self.train_batch_idx = self.train_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, y_batch, batch_end

  def get_val_batch(self, n_batch=None):
    if self._x_val is None and self._y_val is None:
      self._get_dataset()

    if n_batch is None:
      return self._x_val, self._y_val

    batch_end = False

    x_batch = [self.imgs_path + '/' + self._x_val[i]
               for i in list(range(
                 self.val_batch_idx,
                 min(self.val_batch_idx + n_batch,
                     len(self._x_val))))]
    y_batch = [self._y_val[i]
               for i in list(range(
                 self.val_batch_idx,
                 min(self.val_batch_idx + n_batch,
                     len(self._y_val))))]

    if self.val_batch_idx + n_batch < len(self._x_val):
      self.val_batch_idx = self.val_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, y_batch, batch_end

  def get_test_batch(self, n_batch=None):
    if self._x_test is None and self._y_test is None:
      self._get_dataset()

    if n_batch is None:
      return self._x_test, self._y_test

    batch_end = False

    x_batch = [self.imgs_path + '/' + self._x_test[i]
               for i in list(range(
                 self.test_batch_idx,
                 min(self.test_batch_idx + n_batch,
                     len(self._x_test))))]
    y_batch = [self._y_test[i]
               for i in list(range(
                 self.test_batch_idx,
                 min(self.test_batch_idx + n_batch,
                     len(self._y_test))))]

    if self.test_batch_idx + n_batch < len(self._x_test):
      self.test_batch_idx = self.test_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, y_batch, batch_end

  def _get_dataset(self):
    self._x_test = []
    self._y_test = []
    with open(self.anno_path + '/test.txt') as f:
      for line in f:
        img_line = line.rstrip().split()
        self._x_test.append(img_line[0])
        self._y_test.append([int(i) for i in img_line[1:]])

      tup = list(zip(self._x_test, self._y_test))
      random.shuffle(tup)
      self._x_test, self._y_test = zip(*tup)

    self._x_val = []
    self._y_val = []
    with open(self.anno_path + '/val.txt') as f:
      for line in f:
        img_line = line.rstrip().split()
        self._x_val.append(img_line[0])
        self._y_val.append([int(i) for i in img_line[1:]])

      tup = list(zip(self._x_val, self._y_val))
      random.shuffle(tup)
      self._x_val, self._y_val = zip(*tup)

    self._x_train = []
    self._y_train = []
    with open(self.anno_path + '/train.txt') as f:
      for line in f:
        img_line = line.rstrip().split()
        self._x_train.append(img_line[0])
        self._y_train.append([int(i) for i in img_line[1:]])

      tup = list(zip(self._x_train, self._y_train))
      random.shuffle(tup)
      self._x_train, self._y_train = zip(*tup)

  @staticmethod
  def build_dataset_celeba(anno_path='dataset/CelebA/anno',
                           imgs_path='dataset/CelebA/imgs',
                           train_percent=80,
                           val_percent=19,
                           test_percent=1):
    target_labels = {'Black_Hair': 'Black Hair',
                    'Blond_Hair': 'Blond Hair',
                    'Male': 'Male',
                    'Young': 'Young',
                    'Smiling': 'Smile'}
    imgs_name = []
    imgs_labels = []
    with open(anno_path + '/original.txt') as f:
      n_imgs = int(f.readline().rstrip())
      anno_labels = f.readline().rstrip().split()
      anno_labels_dict = {k: v + 1 for v, k in enumerate(anno_labels)}
      for line in f:
        img_line = line.rstrip().split()

        imgs_name.append(img_line[0]) # img[0] - name of the image

        img_labels = [0 for i in list(range(len(Dataset().labels.items())))]
        for k, v in target_labels.items():
          anno_labels_pos = anno_labels_dict[k]
          if int(img_line[anno_labels_pos]) == 1:
            img_labels[Dataset().labels[v]] = 1
          else:
            if k == 'Male':
              img_labels[Dataset().labels['Female']] = 1
            elif k == 'Young':
              img_labels[Dataset().labels['Old']] = 1
        imgs_labels.append(img_labels)
      tup = list(zip(imgs_name, imgs_labels))
      random.shuffle(tup)
      imgs_name, imgs_labels = zip(*tup)

    n_test = int(n_imgs * test_percent / 100) + 1
    n_val = int(n_imgs * val_percent / 100) + 1

    x_test = imgs_name[:n_test]
    y_test = imgs_labels[:n_test]

    x_val = imgs_name[n_test:n_val]
    y_val = imgs_labels[n_test:n_val]

    x_train = imgs_name[n_val:]
    y_train = imgs_labels[n_val:]

    with open(anno_path + '/test.txt', 'w') as f:
      for i in list(range(len(x_test))):
        img_line = x_test[i] + ' ' + ' '.join([str(l) for l in y_test[i]])
        f.write(img_line + '\n')

    with open(anno_path + '/val.txt', 'w') as f:
      for i in list(range(len(x_val))):
        img_line = x_val[i] + ' ' + ' '.join([str(l) for l in y_val[i]])
        f.write(img_line + '\n')

    with open(anno_path + '/train.txt', 'w') as f:
      for i in list(range(len(x_train))):
        img_line = x_train[i] + ' ' + ' '.join([str(l) for l in y_train[i]])
        f.write(img_line + '\n')
