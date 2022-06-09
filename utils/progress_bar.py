from tqdm import tqdm
import logging


def get_progress_bar(iterations: int,
                     mode: str = 'simple',
                     round_base: int = 2,
                     show_steps: bool = False,
                     steps: int = 2,
                     leave: bool = True):
    if mode == 'tqdm':
        return TqdmProgressBar(iterations, round_base, leave)
    elif mode == 'simple':
        return SimpleProgressBar(iterations, round_base, show_steps, steps)
    elif mode == 'logging':
        return LoggingProgressBar(iterations, round_base, show_steps, steps)
    else:
        raise NotImplementedError(f"{mode} progress bar has not implemented!")

class BaseProgressBar(object):

    def __init__(self, iterations, round_base: int = 2):
        self.iterations = iterations
        self.round_base = round_base

    def get_description(self, epoch):
        raise NotImplementedError('No description method has been added!')

    def get_postfix(self, accuracy, loss):
        raise NotImplementedError('No postfix method has been added!')

    def show(self, epoch, loss, accuracy):
        raise NotImplementedError('No method has been added!')


class TqdmProgressBar(BaseProgressBar):

    def __init__(self, iterations, round_base: int = 2, leave: bool = True):
        super(TqdmProgressBar, self).__init__(iterations, round_base)
        self.tqdm = tqdm(
            iterations,
            leave=leave,
        )

        self.round_base = round_base

    def get_description(self, epoch: int):
        return f"Epoch {epoch}"

    def get_postfix(self, accuracy, loss):
        return f"Loss: {round(loss, self.round_base)}, Accuracy: {round(accuracy, self.round_base)}"

    def show(self, epoch, loss, accuracy):
        description = self.get_description(epoch)

        self.tqdm.set_description(description)
        self.tqdm.set_postfix(loss=round(loss, self.round_base),
                              accuracy=round(accuracy, self.round_base) * 100)


class LoggingProgressBar(BaseProgressBar):

    def __init__(self, iterations, round_base: int = 2, show_steps: bool = True,
                 step: int = 2):
        super(LoggingProgressBar, self).__init__(iterations, round_base)

        self.show_steps = show_steps
        self.show_per_steps = iterations / step

    def get_description(self, epoch: int):
        return f"Epoch {epoch}=>"

    def get_postfix(self, accuracy, loss):
        return f"Loss: {round(loss, self.round_base)}, Accuracy: {round(accuracy, self.round_base)}"

    def show(self, epoch, loss, accuracy):
        logging.info(f"{self.get_description(epoch)}{self.get_postfix(loss, accuracy)}")


class SimpleProgressBar(BaseProgressBar):
    def __init__(self, iterations, round_base: int = 2, show_steps: bool = True,
                 step: int = 2):
        super(SimpleProgressBar, self).__init__(iterations, round_base)

        self.show_steps = show_steps
        self.show_per_steps = iterations / step

    def get_description(self, epoch: int):
        return f"Epoch {epoch} => "

    def get_postfix(self, accuracy, loss):
        return f"Loss: {round(loss, self.round_base)}, Accuracy: {round(accuracy, self.round_base)}"

    def show(self, epoch, loss, accuracy):
        print(f"{self.get_description(epoch)}{self.get_postfix(loss, accuracy)}")
