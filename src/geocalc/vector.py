import math

class Vector:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z if z is not None else 0

    def __repr__(self):
        if self.z:
            return f"Vector({self.x}, {self.y}, {self.z})"
        else:
            return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + (self.z * other.z if self.z is not None and other.z is not None else 0)
        elif isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.z * other if self.z is not None else None)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)

    def magnitude(self)->float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalize(self)->'Vector':
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)
    
    def magnitude_squared(self)->float:
        return self.magnitude()**2
    
    def angle_between(self, other_vector)->float:
        dot_product = self.dot(other_vector)
        magnitude_product = self.magnitude() * other_vector.magnitude()
        cos_angle = dot_product / magnitude_product
        angle = math.acos(cos_angle)
        return angle

    def projection(self, other_vector:"Vector")->'Vector':
        projection_coefficient = self.dot(other_vector) / other_vector.magnitude_squared()
        projection_vector = other_vector * projection_coefficient
        return projection_vector
    
    def is_orthogonal(self, other_vector)-> bool:
        """Check if two vectors are orthogonal (perpendicular)"""
        return math.isclose(self.dot(other_vector), 0, rel_tol=1e-9)
    
    def is_parallel(self, other_vector) -> bool:
        cross_product = self.cross(other_vector)
        return cross_product.magnitude() < 1e-9
    
    def lerp(self, other_vector, t):
        """Linear interpolation between two vectors"""
        return (1 - t) * self + t * other_vector




