import argparse

from starGAN import StarGAN

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--train', help='if true, start training a neural network',
                                 action='store_true')
  parser.add_argument('--resume', help='if true, resume training from last checkpoint',
                                  action='store_true')
  parser.add_argument('--predict_test', help='run the neural net on test set',
                                 action='store_true')
  parser.add_argument('--batchSize', help='batch size for training, default=8',
                                     type=int,
                                     default=8)
  args = parser.parse_args()

  if args.train:
    model = StarGAN(batch_size=args.batchSize)
    model.build()

    if args.resume:
      model.train(resume=True)
    else:
      model.train()
  elif args.predict_test:
    model = StarGAN(batch_size=1)
    model.build()

    model.predict_test()
  else:
    print('Nothing to be done!')
