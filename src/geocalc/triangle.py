import math
from typing import Optional, Tuple

class Triangle:
    def __init__(self, a:Optional[float] = None, b:Optional[float] = None, c:Optional[float] = None, angle_a:Optional[float] = None, 
                 angle_b:Optional[float] = None, angle_c:Optional[float] = None, 
                 A:Optional[Tuple[float, float]] = None, B:Optional[Tuple[float, float]] = None, C:Optional[Tuple[float, float]] = None):
        self.a = a
        self.b = b
        self.c = c
        self.angle_a = angle_a
        self.angle_b = angle_b
        self.angle_c = angle_c
        self.A = A
        self.B = B
        self.C = C

        # Compute side lengths if vertices are provided
        if self.A and self.B and self.C:
            self.a = self.distance(self.B, self.C)
            self.b = self.distance(self.A, self.C)
            self.c = self.distance(self.A, self.B)

        # Compute angles if side lengths are provided
        if self.a and self.b and self.c:
            if not self.angle_a:
                self.angle_a = self.angle_from_sides(self.b, self.c, self.a)
            if not self.angle_b:
                self.angle_b = self.angle_from_sides(self.a, self.c, self.b)
            if not self.angle_c:
                self.angle_c = self.angle_from_sides(self.a, self.b, self.c)

    @staticmethod
    def distance(p1, p2) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def angle_from_sides(a:float, b:float, c:float)->float:
        return math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))

    @staticmethod
    def heron_formula(a:float, b:float, c:float)->float:
        """Calculates the area of a triangle using Heron's formula."""
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))

    @staticmethod
    def trigonometric_formula(a:float, b:float, angle_c)->float:
        """Calculates the area of a triangle using the trigonometric formula (1/2 * a * b * sin(C))."""
        angle_c_rad = math.radians(angle_c)
        return 0.5 * a * b * math.sin(angle_c_rad)

    @staticmethod
    def base_height_formula(base:float, height:float)->float:
        """Calculates the area of a triangle using the base-height formula (1/2 * base * height)."""
        return 0.5 * base * height
    
    @staticmethod
    def coordinates_formula(coords: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None, 
                            A: Optional[Tuple[float, float]] = None, B: Optional[Tuple[float, float]] = None, C: Optional[Tuple[float, float]] = None) -> float:
        """Calculates the area of a triangle using the coordinates formula."""
        
        if coords is not None:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
        else:
            if A is None or B is None or C is None:
                raise ValueError("If coords is not provided, A, B, and C must be provided.")
            
            x1, y1 = A 
            x2, y2 = B 
            x3, y3 = C 

        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

    def heron(self) -> float:
        """Calculates the area of a triangle using Heron's formula."""
        
        if self.a is None or self.b is None or self.c is None:
            raise ValueError("All sides (a, b, and c) must be provided to use Heron's formula.")
        
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def trigonometric(self) -> float:
        """Calculates the area of a triangle using the trigonometric formula (1/2 * a * b * sin(C))."""
        if self.a is None or self.b is None or self.angle_c is None:
            raise ValueError("Sides a, b and the angle between them (angle_c) are required for the trigonometric formula.")
        
        angle_c_rad = math.radians(self.angle_c)
        return 0.5 * self.a * self.b * math.sin(angle_c_rad)

    def cosine_rule(self):
        if self.a and self.b and self.angle_c:
            self.c = math.sqrt(self.a**2 + self.b**2 - 2 * self.a * self.b * math.cos(math.radians(self.angle_c)))
        elif self.a and self.c and self.angle_b:
            self.b = math.sqrt(self.a**2 + self.c**2 - 2 * self.a * self.c * math.cos(math.radians(self.angle_b)))
        elif self.b and self.c and self.angle_a:
            self.a = math.sqrt(self.b**2 + self.c**2 - 2 * self.b * self.c * math.cos(math.radians(self.angle_a)))
        else:
            raise ValueError("Not enough information provided for the cosine rule.")

    def sine_rule(self):
        known_sides = [self.a, self.b, self.c].count(None)
        known_angles = [self.angle_a, self.angle_b, self.angle_c].count(None)

        if known_sides + known_angles != 1:
            raise ValueError("The sine rule requires exactly one unknown side or angle.")

        if self.a is not None and self.angle_a is not None:
            ratio = self.a / math.sin(math.radians(self.angle_a))
            if self.b is None and self.angle_b is not None:
                self.b = ratio * math.sin(math.radians(self.angle_b))
            elif self.c is None and self.angle_c is not None:
                self.c = ratio * math.sin(math.radians(self.angle_c))
            elif self.angle_b is None and self.b is not None:
                self.angle_b = math.degrees(math.asin(self.b / ratio))
                self.angle_c = 180 - self.angle_a - self.angle_b
        elif self.b is not None and self.angle_b is not None:
            ratio = self.b / math.sin(math.radians(self.angle_b))
            if self.a is None and self.angle_a is not None:
                self.a = ratio * math.sin(math.radians(self.angle_a))
            elif self.c is None and self.angle_c is not None:
                self.c = ratio * math.sin(math.radians(self.angle_c))
            elif self.angle_a is None and self.a is not None:
                self.angle_a = math.degrees(math.asin(self.a / ratio))
                self.angle_c = 180 - self.angle_a - self.angle_b
        elif self.c is not None and self.angle_c is not None:
            ratio = self.c / math.sin(math.radians(self.angle_c))
            if self.a is None and self.angle_a is not None:
                self.a = ratio * math.sin(math.radians(self.angle_a))
            elif self.b is None and self.angle_b is not None:
                self.b = ratio * math.sin(math.radians(self.angle_b))
            elif self.angle_a is None and self.a is not None:
                self.angle_a = math.degrees(math.asin(self.a / ratio))
                self.angle_b = 180 - self.angle_a - self.angle_c
        else:
            raise ValueError("Not enough information provided for the sine rule.")

    def area(self, method='heron'):
        if method == 'heron':
            return self.heron()
        elif method == 'sine':
            return self.trigonometric()
        else:
            raise ValueError("Invalid area calculation method specified.")

    def calculate_other_coordinates(self):
        coords = []
        if self.A is not None:
            coords.append(self.A)
        if self.B is not None:
            coords.append(self.B)
        if self.C is not None:
            coords.append(self.C)
        if not coords or len(coords) != 2:
            raise ValueError("Two known vertex coordinates are required.")
        if not self.a or not self.b or not self.c:
            raise ValueError("All side lengths are required.")

        x1, y1 = coords[0]
        x2, y2 = coords[1]

        # Calculate the distance between the two known vertices
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Determine which side corresponds to the distance between the known vertices
        if math.isclose(self.a, d):
            side_opposite = self.c
            side_adjacent = self.b
        elif math.isclose(self.b, d):
            side_opposite = self.a
            side_adjacent = self.c
        elif math.isclose(self.c, d):
            side_opposite = self.b
            side_adjacent = self.a
        else:
            raise ValueError("The side lengths provided do not match the distance between the known vertices.")

        # Calculate the coordinates of the third vertex
        cos_theta = (side_adjacent**2 - side_opposite**2 + d**2) / (2 * d * side_adjacent)
        sin_theta = math.sqrt(1 - cos_theta**2)

        x3 = x1 + (x2 - x1) * cos_theta * side_adjacent / d - (y2 - y1) * sin_theta * side_adjacent / d
        y3 = y1 + (x2 - x1) * sin_theta * side_adjacent / d + (y2 - y1) * cos_theta * side_adjacent / d

        # Alternative solution
        x3_alt = x1 + (x2 - x1) * cos_theta * side_adjacent / d + (y2 - y1) * sin_theta * side_adjacent / d
        y3_alt = y1 - (x2 - x1) * sin_theta * side_adjacent / d + (y2 - y1) * cos_theta * side_adjacent / d

        return (x3, y3), (x3_alt, y3_alt)