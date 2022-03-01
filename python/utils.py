from enum import Enum


class MagneticId(Enum):
    HEAD = 2
    SHOULDER_LEFT = 0
    SHOULDER_RIGHT = 1
    HAND_LEFT = 4
    HAND_RIGHT = 3


class BodyPart(str, Enum):
    HEAD = "head"
    SHOULDER_LEFT = "shoulder_left"
    SHOULDER_RIGHT = "shoulder_right"
    HAND_LEFT = "hand_left"
    HAND_RIGHT = "hand_right"
    SHOULDER = "shoulder"

    def __str__(self):
        return self.value

