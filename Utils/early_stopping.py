class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = max(0, patience)
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.num_bad_epochs = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        if self.patience <= 0:
            return False
        if self._is_improvement(score):
            self.best_score = score
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
