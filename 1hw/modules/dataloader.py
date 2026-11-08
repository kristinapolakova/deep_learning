import numpy as np
class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        # число батчей = округление вверх от (число объектов / размер батча)
        # например, 100 объектов и батч 32 — будет 4 батча (3 полных + 1 неполный)
        return (self.X.shape[0] + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        # просто число строк в X
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
            # перемешиваем индексы: получим массив [3, 0, 5, 2, ...] в случайном порядке
            self.indices = np.random.permutation(self.X.shape[0])
        else:
            # идём по порядку: [0, 1, 2, ...]
            self.indices = np.arange(self.X.shape[0])

        # обнуляем счётчик батчей
        self.batch_id = 0
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        # когда все батчи выданы — сигнализируем об этом
        if self.batch_id >= len(self):
            raise StopIteration

        # вычисляем, какие строки взять: от start до end (не включая end)
        start = self.batch_id * self.batch_size
        end = min(start + self.batch_size, self.X.shape[0])

        # индексы этих строк (с учётом перемешивания)
        idx = self.indices[start:end]

        # увеличиваем счётчик батчей
        self.batch_id += 1

        # возвращаем порцию данных
        return self.X[idx], self.y[idx]
