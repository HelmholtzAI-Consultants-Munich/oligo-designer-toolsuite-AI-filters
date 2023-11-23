import abc

class APIBase():

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def predict(self):
        pass

