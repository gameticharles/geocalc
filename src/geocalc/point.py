import math

class Point:
    def __init__(self, x, y, z = None):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        if self.z is not None:
            return f"({self.x}, {self.y}, {self.z})"
        else:
            return f"({self.x}, {self.y})" 

    def __repr__(self):
        return f"Point{self.__str__()}"
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y, self.z - other.z if self.z is not None and other.z is not None else None)
    
    def __mul__(self, scalar: float):
        return Point(self.x * scalar, self.y * scalar, self.z * scalar if self.z is not None else None)
    
    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        return Point(self.x / scalar, self.y / scalar, self.z / scalar if self.z is not None else None)

    def rotate(self, angle_degrees: float):
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        x_rotated = self.x * cos_angle - self.y * sin_angle
        y_rotated = self.x * sin_angle + self.y * cos_angle
        return Point(x_rotated, y_rotated, self.z)
    
    def distance_to(self, other_point:"Point"):
        dx = other_point.x - self.x
        dy = other_point.y - self.y
        if self.z is not None and other_point.z is not None:
            dz = other_point.z - self.z
            return math.sqrt(dx**2 + dy**2 + dz**2)
        else:
            return math.sqrt(dx**2 + dy**2)

    def move(self, dx, dy, dz=None):
        self.x += dx
        self.y += dy
        if dz is not None and self.z is not None:
            self.z += dz

    def rotate_by(self, angle_degrees, origin=None):
        if origin is None:
            origin = Point(0, 0)

        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)

        translated_x = self.x - origin.x
        translated_y = self.y - origin.y

        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle

        self.x = rotated_x + origin.x
        self.y = rotated_y + origin.y


    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        if self.z is not None:
            self.z *= factor

    def midpoint_to(self, other_point:"Point"):
        mx = (self.x + other_point.x) / 2
        my = (self.y + other_point.y) / 2
        if self.z is not None and other_point.z is not None:
            mz = (self.z + other_point.z) / 2
            return Point(mx, my, mz)
        else:
            return Point(mx, my)

    def reflect(self, origin=None):
        if origin is None:
            origin = Point(0, 0, z=0)

        rx = 2 * origin.x - self.x
        ry = 2 * origin.y - self.y
        if self.z is not None and origin.z is not None:
            rz = 2 * origin.z - self.z
            return Point(rx, ry, rz)
        else:
            return Point(rx, ry)

    def angle_between(self, other_point:"Point"):
        if self.z is not None or other_point.z is not None:
            raise ValueError("Angle between is not supported for 3D points.")
        dx = other_point.x - self.x
        dy = other_point.y - self.y
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def as_tuple(self):
        if self.z is not None:
            return (self.x, self.y, self.z)
        else:
            return (self.x, self.y)
    
    def translate(self, dx, dy, dz=None):
        self.x += dx
        self.y += dy
        if dz is not None and self.z is not None:
            self.z += dz
    
    def polar_coordinates(self):
        if self.z is not None:
            raise ValueError("Polar coordinates are not supported for 3D points.")
    
        r = math.sqrt(self.x ** 2 + self.y ** 2)
        theta = math.degrees(math.atan2(self.y, self.x))
        return r, theta
    
    def set_polar_coordinates(self, r, theta, origin=None):
        if origin is None:
            origin = Point(0, 0)

        theta_radians = math.radians(theta)
        self.x = origin.x + r * math.cos(theta_radians)
        self.y = origin.y + r * math.sin(theta_radians)
        
    def slope_to(self, other_point:"Point"):
        dy = other_point.y - self.y
        dx = other_point.x - self.x
        if dx == 0:
            return float('inf') # Vertical line: slope is undefined
        return dy / dx

    def is_collinear(self, a:"Point", b:"Point"):
        return (b.y - a.y) * (b.x - self.x) == (b.y - self.y) * (b.x - a.x)
    
    def distance_to_line(self, line_point1:"Point", line_point2:"Point"):
        num = abs((line_point2.y - line_point1.y) * self.x - (line_point2.x - line_point1.x) * self.y + line_point2.x * line_point1.y - line_point2.y * line_point1.x)
        den = ((line_point2.y - line_point1.y)**2 + (line_point2.x - line_point1.x)**2)**0.5
        return num / den

    def bearing_to(self, point:"Point"):
        angle = math.atan2(point.y - self.y, point.x - self.x)
        bearing = math.degrees(angle) % 360
        return bearing

    def distance_to_circle(self, center:"Point", radius):
        distance_to_center = self.distance_to(center)
        return max(0, distance_to_center - radius)

    def distance_to_polyline(self, polyline):
        min_distance = float("inf")
        for i in range(len(polyline) - 1):
            distance = self.distance_to_line(polyline[i], polyline[i + 1])
            min_distance = min(min_distance, distance)
        return min_distance

    def interpolate(self, point1, point2, ratio):
        x = point1.x + ratio * (point2.x - point1.x)
        y = point1.y + ratio * (point2.y - point1.y)
        if self.z is not None and point1.z is not None and point2.z is not None:
            z = point1.z + ratio * (point2.z - point1.z)
            return Point(x, y, z)
        return Point(x, y)

    def triangle_area(self, point1, point2):
        area = 0.5 * abs((point1.x - self.x) * (point2.y - self.y) - (point1.y - self.y) * (point2.x - self.x))
        return area

    def is_inside_polygon(self, polygon):
        num_vertices = len(polygon)
        j = num_vertices - 1
        inside = False

        for i in range(num_vertices):
            if ((polygon[i].y > self.y) != (polygon[j].y > self.y)) and \
                    (self.x < (polygon[j].x - polygon[i].x) * (self.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
            j = i
        return inside


