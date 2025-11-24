class AlphaScheduler:
    def get_alpha(self) -> float:
        raise NotImplementedError

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
        }


class FixedAlpha(AlphaScheduler):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def get_alpha(self) -> float:
        return self.alpha

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict["alpha"] = self.alpha
        return base_dict
