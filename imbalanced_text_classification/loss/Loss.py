from enum import Enum

# class syntax
class Loss(Enum):
    CE_Loss = "CE_Loss"
    Weighted_CE_Loss = "Weighted_CE_Loss"
    Focal_Loss = "Focal_Loss"
    Weighted_Focal_Loss = "Weighted_Focal_Loss"