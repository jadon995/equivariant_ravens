import os
import sys

class Task():
    def __init__(self) -> None:

        self.mode = 'train'
        self.max_steps = 4
        pass

    def get_reward(self):
        return 1.0 / self.max_steps
    
class KitFourTools(Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = 4

class KitSixTools(Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = 6

class StackBlockPyramid(Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = 6

class PlaceRedInGreen(Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = 1

