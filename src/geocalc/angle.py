import math
from typing import Tuple

class Angle:
    def __init__(self, value =None, unit='deg'):
        if value is not None:
            if unit == 'dms':
                self.dms = value
                self.deg = self.dms_to_degrees(*value)
            elif unit == 'deg':
                self.deg = value
                self.dms = self.degrees_to_dms(value)
            elif unit == 'rad':
                self.deg = self.radians_to_degrees(value)
                self.dms = self.degrees_to_dms(self.deg)
            elif unit == 'grad':
                self.deg = self.grads_to_degrees(value)
                self.dms = self.degrees_to_dms(self.deg)
            else:
                raise ValueError("Invalid unit specified.")

            self.rad = self.degrees_to_radians(self.deg)
            self.grad = self.degrees_to_grads(self.deg)

    def degrees_to_radians(self, degrees:float) -> float:
        return math.radians(degrees)

    def radians_to_degrees(self, radians:float)-> float:
        return math.degrees(radians)

    def degrees_to_grads(self, degrees:float)-> float:
        return degrees * 10/9

    def grads_to_degrees(self, grads:float)-> float:
        return grads * 9/10

    def radians_to_grads(self, radians:float)-> float:
        return self.radians_to_degrees(radians) * 10/9

    def grads_to_radians(self, grads:float)-> float:
        return self.degrees_to_radians(grads * 9/10)

    def degrees_to_dms(self, degrees:float)-> Tuple[int,int ,float]:
        d = int(degrees)
        m = int((degrees - d) * 60)
        s = (degrees - d - m/60) * 3600
        return d, m, s

    def dms_to_degrees(self, d:float, m:float, s:float)-> float:
        return d + m/60 + s/3600
