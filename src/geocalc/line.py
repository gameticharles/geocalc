from .point import Point
from typing import Union
import math

class Line:
    def __init__(self, point1:Point, point2:Point):
        if point1 == point2:
            raise ValueError("The two points must be distinct.")
        self.point1 = point1
        self.point2 = point2

    def __repr__(self):
        return f"Line({self.point1}, {self.point2})"

    def length(self):
        return self.point1.distance_to(self.point2)

    def slope(self)->float:
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def intercept(self)->float:
        return self.point1.y - self.slope() * self.point1.x

    def midpoint(self)->Point:
        return self.point1.midpoint_to(self.point2)

    def contains_point(self, point:"Point")->bool:
        return point.is_collinear(self.point1, self.point2) and self.point1.distance_to(point) + self.point2.distance_to(point) == self.length()

    def perpendicular_bisector(self):
        midpoint = self.midpoint()
        if self.point2.x == self.point1.x:
            return Line(midpoint, Point(midpoint.x + 1, midpoint.y))
        slope = -1 / self.slope()
        intercept = midpoint.y - slope * midpoint.x
        return Line(midpoint, Point(midpoint.x + 1, slope * (midpoint.x + 1) + intercept))

    def angle_between(self, other_line:"Line") -> float:
        tan_angle = abs((self.slope() - other_line.slope()) / (1 + self.slope() * other_line.slope()))
        return math.degrees(math.atan(tan_angle))
    
    def point_at_distance(self, point: Point, distance: float) -> Point:
        if not self.contains_point(point):
            raise ValueError("The given point must be on the line.")
            
        line_length = self.length()
        ratio = distance / line_length

        x = point.x + ratio * (self.point2.x - self.point1.x)
        y = point.y + ratio * (self.point2.y - self.point1.y)

        return Point(x, y)

    def intersection(self, other_line:"Line") -> Union[Point,None]:
        if self.slope() == other_line.slope():
            return None
        x = (other_line.intercept() - self.intercept()) / (self.slope() - other_line.slope())
        y = self.slope() * x + self.intercept()
        return Point(x, y)

    def parallel_to(self, other_line:"Line")-> bool:
        return self.slope() == other_line.slope()

    def perpendicular_to(self, other_line:"Line") -> bool :
        return self.slope() * other_line.slope() == -1

    def is_parallel(self, other_line) -> bool:
        return abs(self.slope() - other_line.slope()) < 1e-9

    def is_perpendicular(self, other_line) -> bool:
        return abs(self.slope() * other_line.slope() + 1) < 1e-9

    def angle_with_x_axis(self) -> float:
        return math.degrees(math.atan(self.slope()))

    def angle_with_another_line(self, other_line)->float:
        if self.is_parallel(other_line):
            return 0.0
        angle = abs(math.atan((other_line.slope() - self.slope()) / (1 + self.slope() * other_line.slope())))
        return math.degrees(angle)

    def as_tuple(self):
        return (self.point1.as_tuple(), self.point2.as_tuple())
