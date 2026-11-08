import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # средний квадрат разницы между предсказанием и правильным ответом
        diff = input - target
        return np.mean(diff ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # производная MSE: 2 * (input - target) / количество элементов
        return 2.0 * (input - target) / input.size


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_probs = self.log_softmax(input)
        B, C = input.shape

        # если smoothing = 0 — обычная кросс-энтропия (быстрый путь)
        if self.label_smoothing == 0.0:
            return -np.mean(log_probs[np.arange(B), target])

        # иначе — строим сглаженные цели и считаем -sum(target_c * log_prob_c)
        # target_smooth[i, c] = (1 - smoothing) * [c == y_i] + smoothing / C
        target_smooth = np.full((B, C), self.label_smoothing / C, dtype=log_probs.dtype)
        target_smooth[np.arange(B), target] += (1.0 - self.label_smoothing)
        # loss = -sum(target_smooth * log_probs) / B
        return -np.sum(target_smooth * log_probs) / B

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        # получаем вероятности из логитов (softmax)
        log_probs = self.log_softmax(input)
        probs = np.exp(log_probs)
        B, C = input.shape

        # если smoothing = 0 — обычная формула: (softmax - one_hot) / B
        if self.label_smoothing == 0.0:
            grad = probs.copy()
            grad[np.arange(B), target] -= 1.0
            return grad / B

        # иначе — вычитаем сглаженные цели:
        # grad = (softmax - target_smooth) / B
        target_smooth = np.full((B, C), self.label_smoothing / C, dtype=log_probs.dtype)
        target_smooth[np.arange(B), target] += (1.0 - self.label_smoothing)
        return (probs - target_smooth) / B
