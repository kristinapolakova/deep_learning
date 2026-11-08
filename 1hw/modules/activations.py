import numpy as np
from .base import Module
from scipy.special import erf, logsumexp

class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # всё, что меньше нуля — заменяем нулём
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # градиент проходит только там, где вход был положительным
        return grad_output * (input > 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # формула: 1 / (1 + e^(-x))
        return 1.0 / (1.0 + np.exp(-input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # производная сигмоиды: s * (1 - s), где s = sigmoid(input)
        s = 1.0 / (1.0 + np.exp(-input))
        return grad_output * s * (1 - s)


class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # y = x * Φ(x), где Φ — функция распределения нормального
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return 0.5 * input * (1.0 + erf(input / np.sqrt(2)))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # производная GELU = Φ(x) + x * φ(x), где φ — плотность нормального
        cdf = 0.5 * (1.0 + erf(input / np.sqrt(2)))
        pdf = np.exp(-input ** 2 / 2) / np.sqrt(2 * np.pi)
        return grad_output * (cdf + input * pdf)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # вычитаем максимум для численной стабильности (это не меняет результат)
        shifted = input - input.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        # делим каждое значение на сумму по строке
        return exp / exp.sum(axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        shifted = input - input.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        s = exp / exp.sum(axis=-1, keepdims=True)
        # формула: s * (grad_output - sum(grad_output * s) по строке)
        return s * (grad_output - (grad_output * s).sum(axis=-1, keepdims=True))


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # log_softmax(x) = x - logsumexp(x)
        # logsumexp — стабильная версия log(sum(exp(x)))
        return input - logsumexp(input, axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # производная log_softmax: grad_output - softmax(x) * sum(grad_output)
        s = np.exp(input - logsumexp(input, axis=-1, keepdims=True))
        return grad_output - s * grad_output.sum(axis=-1, keepdims=True)
