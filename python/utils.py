from enum import Enum


class BodyPart(str, Enum):
    HEAD = "head"
    SHOULDER_LEFT = "shoulder_left"
    SHOULDER_RIGHT = "shoulder_right"
    HAND_LEFT = "hand_left"
    HAND_RIGHT = "hand_right"

    def __str__(self):
        return self.value