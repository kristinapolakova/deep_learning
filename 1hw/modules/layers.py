import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    Применяет линейное (аффинное) преобразование данных: y = x W^ T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features (:параметр in_features: входные векторные характеристики)
        :param out_features: output vector features (параметр out_features: выходные векторные характеристики)
        :param bias: whether to use additive bias (:параметр bias: использовать ли аддитивное смещение)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features) (параметр input: массив фигур (batch_size, in_features))
        :return: array of shape (batch_size, out_features) (возвращает: массив фигур (batch_size, out_features))
        """
        
        # # линейное преобразование
        # return super().compute_output(input)
        y = input @ self.weight.T
        # если есть смещение, то прибавляем его
        if self.bias is not None:
            y = y + self.bias
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        
        # return super().compute_grad_input(input, grad_output)
        # производная по входу = grad_output @ W grad_output, умноженный на веса
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        
        # super().update_grad_parameters(input, grad_output)
        # произведение транспонированного grad_output на вход
        self.grad_weight += grad_output.T @ input
        # то же самое для сдвига — если он есть
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # среднее по каждому признаку (по всем объектам в батче)
            self.mean = input.mean(axis=0)
            self.input_mean = input - self.mean
            # дисперсия — среднее от квадратов отклонений
            self.var = (self.input_mean ** 2).mean(axis=0)
            self.sqrt_var = np.sqrt(self.var + self.eps)
            # нормированный вход
            self.norm_input = self.input_mean / self.sqrt_var
            # обновляем накопленные статистики для eval-режима
            B = input.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var * B / (B - 1)
        else:
            # в eval используем накопленные статистики
            self.norm_input = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        if self.affine:
            return self.norm_input * self.weight + self.bias
        return self.norm_input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if not self.training:
            # градиент: y = (x - running_mean) / sqrt(running_var + eps)
            grad_norm = grad_output * self.weight if self.affine else grad_output
            return grad_norm / np.sqrt(self.running_var + self.eps)
        # в train формула с учётом нормировки по статистикам батча
        B = input.shape[0]
        grad_norm = grad_output * self.weight if self.affine else grad_output
        grad_input = (1.0 / (B * self.sqrt_var)) * (
            B * grad_norm
            - grad_norm.sum(axis=0)
            - self.norm_input * (grad_norm * self.norm_input).sum(axis=0)
        )
        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            # градиент по weight - сумма по батчу от (grad_output * norm_input)
            self.grad_weight += (grad_output * self.norm_input).sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            # генерируем маску: 1 — оставить нейрон, 0 — занулить
            self.mask = (np.random.rand(*input.shape) >= self.p).astype(input.dtype)
            # умножаем на маску и делим на (1 - p), чтобы сохранить средний уровень
            return self.mask * input / (1 - self.p)
        # в eval ничего не делаем — просто возвращаем вход
        return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            # тот же путь, что и в прямом проходе: умножаем на ту же маску и делим на (1 - p)
            return self.mask * grad_output / (1 - self.p)
        # в eval градиент проходит как есть
        return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        
        # сохраняем вход каждого слоя — понадобится при обратном проходе
        self.inputs = [input]
        # прогоняем данные через все слои по очереди
        for module in self.modules:
            input = module(input)
            self.inputs.append(input)
        return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        
        # идём по слоям с конца в начало
        for i in range(len(self.modules) - 1, -1, -1):
            module = self.modules[i]
            # вход этого слоя = self.inputs[i]
            module_input = self.inputs[i]
            # вызываем backward у модуля — он вернёт градиент по своему входу
            # и заодно обновит градиенты своих параметров
            grad_output = module.backward(module_input, grad_output)
        return grad_output

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
