from abc import ABC, abstractmethod

class CharacteristicClassifier(ABC):

    @abstractmethod
    def predict(self, *args, **kwargs) -> float:
        pass

    def __call__(self, *args, **kwds) -> float:
        return self.predict(*args,**kwds)
    
     