import math
import numpy as np
import random
from .point import Point
from .line import Line

class Circle:
    def __init__(self, center, radius):
        if isinstance(center, Point):
            self.center = center
        elif isinstance(center, (tuple, list)) and len(center) == 2:
            self.center = Point(center[0], center[1])
        else:
            raise ValueError("Center must be a Point object or a tuple/list of two coordinates (x, y).")
        self.radius = radius

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def circumference(self)->float:
        return 2 * math.pi * self.radius

    def is_point_inside(self, point)->bool:
        if isinstance(point, Point):
            distance = self.center.distance_to(point)
        elif isinstance(point, (tuple, list)) and len(point) == 2:
            distance = self.center.distance_to(Point(point[0], point[1]))
        else:
            raise ValueError("Point must be a Point object or a tuple/list of two coordinates (x, y).")
        return distance <= self.radius

    def scale(self, factor):
        self.radius *= factor

    def move(self, dx, dy):
        self.center.translate(dx, dy)

    def is_intersecting(self, other_circle)->bool:
        if not isinstance(other_circle, Circle):
            raise ValueError("The argument must be a Circle object.")
        distance_between_centers = self.center.distance_to(other_circle.center)
        return distance_between_centers <= (self.radius + other_circle.radius)

    def intersection_points(self, other_circle:'Circle'):
        if not self.is_intersecting(other_circle):
            return None
        
        d = self.center.distance_to(other_circle.center)
        a = (self.radius ** 2 - other_circle.radius ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(self.radius ** 2 - a ** 2)
        p = self.center + (a / d) * (other_circle.center - self.center)
        p1 = p + h * (other_circle.center - self.center).rotate(90) / d
        p2 = p - h * (other_circle.center - self.center).rotate(90) / d
        return p1, p2
    
    def arc_length(self, point1, point2, direction='ccw') -> float:
        if not (self.is_point_inside(point1) and self.is_point_inside(point2)):
            raise ValueError("Both points must lie on the circle.")
        angle1 = math.atan2(point1.y - self.center.y, point1.x - self.center.x)
        angle2 = math.atan2(point2.y - self.center.y, point2.x - self.center.x)
        
        if direction == 'ccw':
            if angle2 <= angle1:
                angle2 += 2 * math.pi
            arc_angle = angle2 - angle1
        elif direction == 'cw':
            if angle1 <= angle2:
                angle1 += 2 * math.pi
            arc_angle = angle1 - angle2
        else:
            raise ValueError("Direction must be either 'ccw' or 'cw'.")
        
        return self.radius * arc_angle
    
    def tangent_lines(self, point):
        if self.is_point_inside(point):
            raise ValueError("The point must be outside the circle.")
        point_to_center = point - self.center
        distance = point_to_center.magnitude()
        angle = np.arccos(self.radius / distance)
        line_angle = np.arctan2(point_to_center.y, point_to_center.x)
        angle1 = line_angle - angle
        angle2 = line_angle + angle
        return Line(point, angle1), Line(point, angle2)

    def tangent_circles(self, radius, point_on_circle):
        if not self.is_point_inside(point_on_circle):
            raise ValueError("The point must be on the circle.")
        midpoint = self.center.midpoint_to(point_on_circle)
        circle = Circle(midpoint, radius)
        tangent_lines = circle.tangent_lines(self.center)
        
        tangent_circles = []
        for line in tangent_lines:
            point_on_line = line.point_at_distance(circle.center, radius)
            tangent_circles.append(Circle(point_on_line, radius))
        
        return tangent_circles

    
    def sector_area(self, point1, point2, direction='ccw'):
        arc_len = self.arc_length(point1, point2, direction)
        angle = arc_len / self.radius
        return 0.5 * self.radius ** 2 * (angle - np.sin(angle))
    
    def intersecting_area(self, other_circle):
        if not isinstance(other_circle, Circle):
            raise ValueError("The argument must be a Circle object.")

        d = self.center.distance_to(other_circle.center)
        if d >= self.radius + other_circle.radius:
            return 0.0
        if d <= abs(self.radius - other_circle.radius):
            return min(self.area(), other_circle.area())

        r1, r2 = self.radius, other_circle.radius
        alpha = 2 * np.arccos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
        beta = 2 * np.arccos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
        sector_area1 = 0.5 * r1 ** 2 * (alpha - np.sin(alpha))
        sector_area2 = 0.5 * r2 ** 2 * (beta - np.sin(beta))
        return sector_area1 + sector_area2
    
    def common_tangents(self, other_circle):
        if not isinstance(other_circle, Circle):
            raise ValueError("The argument must be a Circle object.")
        
        d = self.center.distance_to(other_circle.center)
        if d == 0:
            return []

        r1, r2 = self.radius, other_circle.radius
        a = (r1 - r2) / d
        b = np.sqrt(1 - a * a)
        p = (other_circle.center - self.center) * a + self.center
        directions = [(b, -a), (-b, a)]
        result = []
        for dx, dy in directions:
            line = Line(Point(p.x + dx * r1, p.y + dy * r1), Point(p.x + dx * r2, p.y + dy * r2))
            result.append(line)
        return result
    
    @classmethod
    def from_boundary_points(cls, boundary_points):
        if len(boundary_points) == 0:
            return cls(Point(0, 0), 0)
        elif len(boundary_points) == 1:
            return cls(boundary_points[0], 0)
        elif len(boundary_points) == 2:
            center = boundary_points[0].midpoint_to(boundary_points[1])
            radius = boundary_points[0].distance_to(center)
            return cls(center, radius)
        else:  # len(boundary_points) == 3
            p1, p2, p3 = boundary_points

            # Circumcenter formula
            D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
            Ux = (
                (p1.x ** 2 + p1.y ** 2) * (p2.y - p3.y)
                + (p2.x ** 2 + p2.y ** 2) * (p3.y - p1.y)
                + (p3.x ** 2 + p3.y ** 2) * (p1.y - p2.y)
            ) / D
            Uy = (
                (p1.x ** 2 + p1.y ** 2) * (p3.x - p2.x)
                + (p2.x ** 2 + p2.y ** 2) * (p1.x - p3.x)
                + (p3.x ** 2 + p3.y ** 2) * (p2.x - p1.x)
            ) / D
            center = Point(Ux, Uy)
            radius = p1.distance_to(center)

            return cls(center, radius)
    
    @staticmethod
    def _welzl_algorithm(points, boundary_points):
        if len(boundary_points) == 3 or len(points) == 0:
            return Circle.from_boundary_points(boundary_points)

        index = random.randint(0, len(points) - 1)
        point = points[index]
        points.pop(index)
        circle = Circle._welzl_algorithm(points, boundary_points)

        if not circle.is_point_inside(point):
            boundary_points.append(point)
            circle = Circle._welzl_algorithm(points, boundary_points)
            boundary_points.pop()

        points.append(point)
        return circle

    @classmethod
    def enclosing_circle(cls, points):
        points = [Point(p[0], p[1]) for p in points]
        random.shuffle(points)
        return cls._welzl_algorithm(points, [])

