import math
import random
import numpy as np
from typing import Optional, Tuple


class Polygon:
    """
    A class representing a polygon, which can be used to compute various properties
    such as area, perimeter, interior and exterior angles, as well as to check
    whether a point is inside the polygon.

    The class can handle both regular and irregular polygons, defined by either
    their vertices or the number of sides and side length.

    Parameters:
    -----------
    vertices : list of tuples or array-like, optional
        A list of (x, y) coordinates defining the vertices of the polygon.
        If not provided, the polygon will be assumed to be regular, and
        the number of sides and side length must be given.
    num_sides : int, optional
        The number of sides in the polygon. Required if vertices are not provided.
    side_length : float, optional
        The length of each side of the polygon. Required for regular polygons.

    Attributes:
    -----------
    vertices : numpy.ndarray
        An array containing the (x, y) coordinates of the polygon vertices.
    sides : list
        A list containing the lengths of the polygon sides.
    num_sides : int
        The number of sides in the polygon.
    side_length : float
        The length of each side of the polygon (if the polygon is regular).

    Example:
    --------
    >>> vertices = [(0, 0), (2, 0), (1, 2)]
    >>> polygon = Polygon(vertices=vertices)
    >>> print(f"Number of sides: {polygon.num_sides}")
    """
    
    def __init__(self, vertices:Optional[list[Tuple[float, float]]]= None, num_sides:Optional[int]=None, side_length:Optional[float]=None): 
        """
        Initialize the Polygon class with either vertices or number of sides and side length.
        """
        self.sides = []
        self.side_length=0 
        self.num_sides =0 
         
        if vertices is not None:
            self.vertices = np.array(vertices)
            self.num_sides = len(vertices)
        elif num_sides is not None:
            self.num_sides = num_sides
        else:
            raise ValueError("Either vertices or num_sides must be provided.")
        
        if side_length is not None:
            self.side_length = side_length
            self.sides = [side_length] * self.num_sides

    def shoelace(self):
        """
        Calculate the area of the polygon using the Shoelace Formula (also known as Gauss's area formula).

        The Shoelace Formula is a simple and efficient method for calculating the area of a polygon
        when its vertices are known. It is particularly well-suited for planar polygons with ordered vertices.
        
        formula:
        A = (1/2) * |Σ(i=1 to n-1)[(xi * yi+1) + (xn * y1)] - Σ(i=1 to n-1)[(xi+1 * yi) + (x1 * yn)]|

        Returns:
        --------
        float
            The area of the polygon.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> area = polygon.shoelace()
        >>> print(f"Area of the polygon: {area:.6f}")
        """
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def greens_theorem(self):
        """
        Calculate the area of the polygon using Green's Theorem.

        Green's Theorem relates a line integral around a simple closed curve to a double integral
        over the plane region it encloses. In the context of a polygon, Green's Theorem can be
        used to compute the area by considering the vertices of the polygon.
        
        formula:
        A = (1/2) * |Σ(i=1 to n-1)[(xi * dyi+1) - (xi+1 * dyi)] + (xn * dy1) - (x1 * dyn)|

        Returns:
        --------
        float
            The area of the polygon.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> area = polygon.greens_theorem()
        >>> print(f"Area of the polygon: {area:.6f}")
        """
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    def triangulation(self):
        """
        Calculate the area of the polygon using triangulation.

        Triangulation is a technique that divides a polygon into a set of non-overlapping triangles.
        The area of the polygon can be computed by summing the areas of all the triangles.
        This method assumes the polygon is simple and non-self-intersecting.
        
        For a simple polygon, break it into triangles and sum their areas. For a triangle with 
        vertices A(x1, y1), B(x2, y2), and C(x3, y3), the area can be computed using the cross-product method:
        Area = (1/2) * |(x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))|

        Returns:
        --------
        float
            The area of the polygon.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> area = polygon.triangulation()
        >>> print(f"Area of the polygon: {area:.6f}")
        """
        area = 0
        for i in range(2, len(self.vertices)):
            area += self.triangle_area(self.vertices[0], self.vertices[i-1], self.vertices[i])
        return area

    def trapezoidal(self):
        """
        Calculate the area of the polygon using the Trapezoidal Rule.

        The Trapezoidal Rule is a numerical integration technique that uses linear
        interpolation to approximate the area enclosed by the vertices of the polygon.
        In the context of a polygon, the Trapezoidal Rule computes the area by summing
        the trapezoidal areas formed by consecutive vertices and the x-axis.
        
        For a polygon with vertices (x1, y1), (x2, y2), ..., (xn, yn) sorted in counter-clockwise order, 
        the area can be approximated as:
        A = (1/2) * Σ(i=1 to n-1)[(xi + xi+1) * (yi+1 - yi)] + (1/2) * (xn + x1) * (y1 - yn)

        Returns:
        --------
        float
            The area of the polygon.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> area = polygon.trapezoidal_rule()
        >>> print(f"Area of the polygon: {area:.6f}")
        """
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * np.sum((np.roll(x, -1) + x) * (np.roll(y, -1) - y))

    def monte_carlo(self, num_points=10000):
        """
        Calculate the area of the polygon using the Monte Carlo method.

        The Monte Carlo method is a statistical technique that uses random sampling to approximate
        the area of a polygon. This method generates a series of random points within the bounding
        rectangle of the polygon and calculates the ratio of points inside the polygon to the total
        number of points. This ratio is then used to estimate the area of the polygon.

        Parameters:
        -----------
        num_points : int, optional, default: 10000
            The number of random points to generate for the Monte Carlo method.
            Increasing the number of points can improve the accuracy of the approximation
            but also increases the computation time.

        Returns:
        --------
        float
            The approximate area of the polygon.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> area = polygon.monte_carlo(num_points=10000)
        >>> print(f"Area of the polygon (Monte Carlo approximation): {area:.6f}")
        """
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])

        inside_points = 0
        for _ in range(num_points):
            x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            if self.point_inside_polygon(x, y):
                inside_points += 1

        rectangle_area = (max_x - min_x) * (max_y - min_y)
        return (inside_points / num_points) * rectangle_area

    @staticmethod
    def triangle_area(a:Tuple[float,float], b:Tuple[float,float], c:Tuple[float,float])->float:
        """
        Calculate the area of a triangle given its vertices.

        This method uses the determinant of a matrix formed by the coordinates of the triangle's vertices
        to compute the area. It can be used as a helper function for other area calculation methods.

        Parameters:
        -----------
        a, b, c : tuple or array-like
            The coordinates (x, y) of the triangle's vertices.

        Returns:
        --------
        float
            The area of the triangle.

        Example:
        --------
        >>> a = (0, 0)
        >>> b = (2, 0)
        >>> c = (1, 2)
        >>> area = Polygon.triangle_area(a, b, c)
        >>> print(f"Area of the triangle: {area:.6f}")
        """
        return 0.5 * np.abs((a[0] * (b[1] - c[1])) + (b[0] * (c[1] - a[1])) + (c[0] * (a[1] - b[1])))

    @staticmethod
    def simpsons_rule(func, a, b, n)->float:
        """
        Approximate the definite integral of a function using Simpson's Rule.

        Simpson's Rule is a numerical integration technique that uses quadratic
        polynomials to approximate the area under the curve of a function.
        
        Parameters:
        -----------
        func : callable
            The function to be integrated. It should accept a single argument (a scalar or a NumPy array)
            and return the corresponding function value(s).
        a : float
            The lower limit of integration.
        b : float
            The upper limit of integration.
        n : int
            The number of subintervals to divide the integration range. Must be an even number.
            A higher value of n increases the accuracy of the approximation but also increases the computation time.

        Returns:
        --------
        float
            The approximate value of the definite integral of the function over the specified range.

        Raises:
        -------
        ValueError
            If n is not an even number.

        Example:
        --------
        >>> def parabola(x):
        ...     return x**2
        ...
        >>> a = 0
        >>> b = 2
        >>> n = 100
        >>> area = simpsons_rule(parabola, a, b, n)
        >>> print(f"Approximate area under the curve y = x^2 between x = {a} and x = {b}: {area:.6f}")
        """
        if n % 2 != 0:
            raise ValueError("n must be an even number.")

        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)

        result = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
        result *= h / 3

        return result

    @staticmethod
    def trapezoidal_rule(func, a:float, b:float, n:int) ->float:
        """
        Approximate the definite integral of a function using the Trapezoidal Rule.

        The Trapezoidal Rule is a numerical integration technique that uses linear
        interpolation to approximate the area under the curve of a function.
        
        Parameters:
        -----------
        func : callable
            The function to be integrated. It should accept a single argument (a scalar or a NumPy array)
            and return the corresponding function value(s).
        a : float
            The lower limit of integration.
        b : float
            The upper limit of integration.
        n : int
            The number of subintervals to divide the integration range. A higher value of n
            increases the accuracy of the approximation but also increases the computation time.

        Returns:
        --------
        float
            The approximate value of the definite integral of the function over the specified range.

        Example:
        --------
        >>> def parabola(x):
        ...     return x**2
        ...
        >>> a = 0
        >>> b = 2
        >>> n = 100
        >>> area = trapezoidal_rule(parabola, a, b, n)
        >>> print(f"Approximate area under the curve y = x^2 between x = {a} and x = {b}: {area:.6f}")
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)

        result = y[0] + y[-1] + 2 * np.sum(y[1:-1])
        result *= h / 2

        return result

    def point_inside_polygon(self, x:float, y:float)->bool:
        """
        Determine if a point is inside the polygon.

        This method uses the ray casting algorithm to check if a point is inside the polygon.
        It casts a ray from the point horizontally to the right and counts the number of
        intersections with the polygon's edges. If the number of intersections is odd,
        the point is inside the polygon; if it's even, the point is outside.

        Parameters:
        -----------
        x, y : float
            The coordinates (x, y) of the point to be tested.

        Returns:
        --------
        bool
            True if the point is inside the polygon, False otherwise.

        Example:
        --------
        >>> vertices = np.array([(0, 0), (2, 0), (1, 2)])
        >>> polygon = Polygon(vertices=vertices)
        >>> x, y = 1, 1
        >>> is_inside = polygon.point_inside_polygon(x, y)
        >>> print(f"Is the point ({x}, {y}) inside the polygon? {is_inside}")
        """
        n = len(self.vertices)
        inside = False
        p1x, p1y = self.vertices[0]
        xinters = 0
        for i in range(n+1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def area_polygon(self):
        """
        Calculate the area of a regular polygon given the number of sides and the side length.

        This method uses the formula for calculating the area of a regular polygon based on its number
        of sides and side length. It requires that the polygon is regular, meaning all sides and angles
        are equal.

        Returns:
        --------
        float
            The area of the regular polygon.

        Raises:
        -------
        ValueError
            If the number of sides or side length is not provided.

        Example:
        --------
        >>> num_sides = 6
        >>> side_length = 2
        >>> polygon = Polygon(num_sides=num_sides, side_length=side_length)
        >>> area = polygon.area_polygon()
        >>> print(f"Area of the regular polygon: {area:.6f}")
        """
        if self.num_sides is None or self.side_length is None:
            raise ValueError("Number of sides and side length are required for regular polygon area calculation.")
        
        return (self.num_sides * self.side_length ** 2) / (4 * math.tan(math.pi / self.num_sides))

    def perimeter(self):
        """
        Calculate the perimeter of a polygon.

        For regular polygons, the perimeter is the product of the number of sides and the side length.
        For irregular polygons, the perimeter is the sum of the side lengths.

        Returns:
        --------
        float
            The perimeter of the polygon.

        Raises:
        -------
        ValueError
            If the number of sides or side length is not provided for a regular polygon.

        Example:
        --------
        >>> vertices = [(0, 0), (2, 0), (1, 2)]
        >>> polygon = Polygon(vertices=vertices)
        >>> perimeter = polygon.perimeter()
        >>> print(f"Perimeter of the polygon: {perimeter:.6f}")
        """
        if self.sides is None:
            raise ValueError("Number of sides and side length are required for regular polygon perimeter calculation.")
        
        return sum(self.sides)

    def interior_angle(self):
        """
        Calculate the interior angle of a regular polygon.

        This method uses the formula for calculating the interior angle of a regular polygon
        based on its number of sides. It requires that the polygon is regular, meaning all
        sides and angles are equal.

        Returns:
        --------
        float
            The interior angle of the regular polygon in degrees.

        Raises:
        -------
        ValueError
            If the number of sides is not provided.

        Example:
        --------
        >>> num_sides = 6
        >>> polygon = Polygon(num_sides=num_sides)
        >>> interior_angle = polygon.interior_angle()
        >>> print(f"Interior angle of the regular polygon: {interior_angle:.2f}°")
        """
        if self.num_sides is None:
            raise ValueError("Number of sides is required for regular polygon interior angle calculation.")
        
        return (self.num_sides - 2) * 180 / self.num_sides

    def exterior_angle(self):
        """
        Calculate the exterior angle of a regular polygon.

        This method uses the formula for calculating the exterior angle of a regular polygon
        based on its number of sides. It requires that the polygon is regular, meaning all
        sides and angles are equal.

        Returns:
        --------
        float
            The exterior angle of the regular polygon in degrees.

        Raises:
        -------
        ValueError
            If the number of sides is not provided.

        Example:
        --------
        >>> num_sides = 6
        >>> polygon = Polygon(num_sides=num_sides)
        >>> exterior_angle = polygon.exterior_angle()
        >>> print(f"Exterior angle of the regular polygon: {exterior_angle:.2f}°")
        """
        if self.num_sides is None:
            raise ValueError("Number of sides is required for regular polygon exterior angle calculation.")
        
        return 360 / self.num_sides
    
    def get_side_length(self, perimeter:Optional[float]=None, area:Optional[float]=None, circumradius:Optional[float]=None, inradius:Optional[float]=None):
        """
        Calculate the side length of a regular polygon given at least one of the parameters
        (perimeter, area, circumradius, inradius).

        Parameters:
        -----------
        perimeter: float, optional
            The perimeter of the regular polygon.

        area: float, optional
            The area of the regular polygon.

        circumradius: float, optional
            The radius of the circumscribed circle (circle that passes through all vertices).

        inradius: float, optional
            The radius of the inscribed circle (circle tangent to all sides).

        Returns:
        --------
        float
            The side length of the regular polygon.

        Raises:
        -------
        ValueError
            If none of the parameters (perimeter, area, circumradius, inradius) are provided.

        Example:
        --------
        >>> num_sides = 6
        >>> polygon = Polygon(num_sides=num_sides)
        >>> side_length = polygon.get_side_length(circumradius=5)
        >>> print(f"Side length of the regular polygon: {side_length:.2f}")
        """
        if perimeter is not None:
            self.side_length = perimeter / self.num_sides
        elif area is not None:
            self.side_length = math.sqrt((4 * area) / (self.num_sides * (1 / math.tan(math.pi / self.num_sides))))
        elif circumradius is not None:
            self.side_length = 2 * circumradius * math.sin(math.pi / self.num_sides)
        elif inradius is not None:
            self.side_length = 2 * inradius * math.tan(math.pi / self.num_sides)
        else:
            raise ValueError("At least one of the parameters (perimeter, area, circumradius, inradius) must be provided.")

        # Update the sides attribute
        self.sides = [self.side_length] * self.num_sides

        return self.side_length
    
    def inradius(self):
        """
        Calculate the inradius of a regular polygon.

        This method uses the formula for calculating the inradius (radius of the inscribed circle)
        of a regular polygon based on its number of sides and side length. It requires that the
        polygon is regular, meaning all sides and angles are equal.

        Returns:
        --------
        float
            The inradius of the regular polygon.

        Raises:
        -------
        ValueError
            If the number of sides and side length are not provided.

        Example:
        --------
        >>> num_sides = 6
        >>> side_length = 5
        >>> polygon = Polygon(num_sides=num_sides, side_length=side_length)
        >>> inradius = polygon.inradius()
        >>> print(f"Inradius of the regular polygon: {inradius:.2f}")
        """
        if self.sides is None:
            raise ValueError("Number of sides and side length are required for regular polygon inradius calculation.")
        
        return self.side_length / (2 * math.tan(math.pi / self.num_sides))

    def circumradius(self):
        """
        Calculate the circumradius of a regular polygon.

        This method uses the formula for calculating the circumradius (radius of the circumscribed circle)
        of a regular polygon based on its number of sides and side length. It requires that the
        polygon is regular, meaning all sides and angles are equal.

        Returns:
        --------
        float
            The circumradius of the regular polygon.

        Raises:
        -------
        ValueError
            If the number of sides and side length are not provided.

        Example:
        --------
        >>> num_sides = 6
        >>> side_length = 5
        >>> polygon = Polygon(num_sides=num_sides, side_length=side_length)
        >>> circumradius = polygon.circumradius()
        >>> print(f"Circumradius of the regular polygon: {circumradius:.2f}")
        """
        if self.sides is None:
            raise ValueError("Number of sides and side length are required for regular polygon circumradius calculation.")
        
        return self.side_length / (2 * math.sin(math.pi / self.num_sides))

    def centroid(self):
        """
        Calculate the centroid of a polygon.

        The centroid is the geometric center of a polygon. It is the arithmetic mean of
        all the vertices' coordinates.

        Returns:
        --------
        numpy.ndarray
            The centroid of the polygon as an array of [x, y] coordinates.

        Example:
        --------
        >>> vertices = [(0, 0), (5, 0), (5, 5), (0, 5)]
        >>> polygon = Polygon(vertices=vertices)
        >>> centroid = polygon.centroid()
        >>> print(f"Centroid of the polygon: {centroid}")
        """
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        centroid_x = np.sum(x_coords) / self.num_sides
        centroid_y = np.sum(y_coords) / self.num_sides
        return np.array([centroid_x, centroid_y])

    def moment_of_inertia(self):
        """
        Calculate the moment of inertia of a polygon.

        The moment of inertia is a measure of the resistance of an object to rotational motion.
        This method calculates the moment of inertia of a polygon about an axis perpendicular
        to its plane and passing through its centroid. The method assumes a uniform mass
        distribution.

        Returns:
        --------
        float
            The moment of inertia of the polygon.

        Example:
        --------
        >>> vertices = [(0, 0), (5, 0), (5, 5), (0, 5)]
        >>> polygon = Polygon(vertices=vertices)
        >>> moment_of_inertia = polygon.moment_of_inertia()
        >>> print(f"Moment of Inertia of the polygon: {moment_of_inertia:.2f}")
        """
        I = 0
        for i in range(self.num_sides):
            j = (i + 1) % self.num_sides
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            I += ((xi ** 2 + xi * xj + xj ** 2) * (yi * xj - xi * yj)) / 12
        return abs(I)

    def is_convex(self):
        """
        Determine if the polygon is convex.

        A polygon is convex if all its interior angles are less than or equal to 180 degrees.
        This method calculates the sum of the angles between consecutive edges and checks if
        the sum is greater than zero.

        Returns:
        --------
        bool
            True if the polygon is convex, False otherwise.

        Example:
        --------
        >>> vertices = [(0, 0), (5, 0), (5, 5), (0, 5)]
        >>> polygon = Polygon(vertices=vertices)
        >>> convex = polygon.is_convex()
        >>> print(f"Is the polygon convex? {convex}")
        """
        angle_sum = 0
        for i in range(self.num_sides):
            prev_vertex = self.vertices[i - 1]
            current_vertex = self.vertices[i]
            next_vertex = self.vertices[(i + 1) % self.num_sides]
            angle_sum += self._angle_between(prev_vertex - current_vertex, next_vertex - current_vertex)
        
        return angle_sum > 0


    def is_point_inside_polygon(self, point)->bool:
        point = np.array(point)
        crossing_count = 0
        for i in range(self.num_sides):
            j = (i + 1) % self.num_sides
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > point[1]) != (yj > point[1])) and (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi):
                crossing_count += 1
        return crossing_count % 2 ==1
    
    def scale(self, factor:float):
        self.vertices *= factor
        if self.sides is not None:
            self.sides = [side * factor for side in self.sides]
            self.side_length *= factor

    def translate(self, dx:float, dy:float):
        translation_vector = np.array([dx, dy])
        self.vertices += translation_vector

    def distance_to_polygon(self, other_polygon):
        min_distance = float('inf')
        for i in range(self.num_sides):
            for j in range(other_polygon.num_sides):
                distance = self._line_segment_distance(self.vertices[i], self.vertices[(i + 1) % self.num_sides],
                                                    other_polygon.vertices[j],
                                                    other_polygon.vertices[(j + 1) % other_polygon.num_sides])
                min_distance = min(min_distance, distance)
        return min_distance

    def side_length_irregular(self, i):
        j = (i + 1) % self.num_sides
        return np.linalg.norm(self.vertices[i] - self.vertices[j])
    
    @staticmethod
    def _angle_between(v1, v2):
        """
        Calculate the angle between two vectors.

        This is a private method used internally by the class to calculate the angle between
        two vectors v1 and v2. The method uses the dot product and the norms of the vectors
        to find the cosine of the angle, and then applies arccos to get the angle in radians.

        Parameters:
        -----------
        v1 : numpy array
            The first vector.
        v2 : numpy array
            The second vector.

        Returns:
        --------
        float
            The angle between v1 and v2 in radians.

        Example:
        --------
        This method is not meant to be called directly by users, but it is used internally
        by other methods like `is_convex`.
        """
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1, 1))

    def angle_between_sides(self, i, j):
        """
        Calculate the angle between two sides of the polygon.

        This method finds the angle between the sides specified by indices i and j.
        The indices are taken modulo the number of sides, so they can be negative or larger than the number of sides.

        Parameters:
        -----------
        i : int
            The index of the first side.
        j : int
            The index of the second side.

        Returns:
        --------
        float
            The angle between the two sides in radians.

        Raises:
        -------
        ValueError
            If the indices i and j are equal.

        Example:
        --------
        >>> polygon = Polygon(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> angle = polygon.angle_between_sides(0, 2)
        """
        if i == j:
            raise ValueError("Indices must be different.")
        i %= self.num_sides
        j %= self.num_sides
        v1 = self.vertices[i] - self.vertices[(i - 1) % self.num_sides]
        v2 = self.vertices[j] - self.vertices[(j - 1) % self.num_sides]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    def bounding_box(self):
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return np.array([min_coords, [max_coords[0], min_coords[1]], max_coords, [min_coords[0], max_coords[1]]])
    
    @staticmethod
    def _line_segment_distance(p1, p2, q1, q2):
        def _point_to_line_distance(p, q1, q2):
            t = np.dot(p - q1, q2 - q1) / np.dot(q2 - q1, q2 - q1)
            if 0 <= t <= 1:
                return float(np.linalg.norm(p - (q1 + t * (q2 - q1))))
            return min(float(np.linalg.norm(p - q1)), float(np.linalg.norm(p - q2)))

        return min(_point_to_line_distance(p1, q1, q2), _point_to_line_distance(p2, q1, q2),
                _point_to_line_distance(q1, p1, p2), _point_to_line_distance(q2, p1, p2))

    # def is_self_intersecting(self):
    #     polygon = ShapelyPolygon(self.vertices)
    #     return not polygon.is_valid

    # def intersection(self, other_polygon):
    #     poly1 = ShapelyPolygon(self.vertices)
    #     poly2 = ShapelyPolygon(other_polygon.vertices)
    #     intersection = poly1.intersection(poly2)
    #     if intersection.is_empty:
    #         return None
    #     return Polygon(vertices=np.array(intersection.exterior.coords[:-1]))

    # def union(self, other_polygon):
    #     poly1 = ShapelyPolygon(self.vertices)
    #     poly2 = ShapelyPolygon(other_polygon.vertices)
    #     union = cascaded_union([poly1, poly2])
    #     if union.geom_type == 'Polygon':
    #         return Polygon(vertices=np.array(union.exterior.coords[:-1]))
    #     return [Polygon(vertices=np.array(poly.exterior.coords[:-1])) for poly in union]

    def nearest_point_on_polygon(self, point):
        point = np.array(point)
        min_distance = float('inf')
        nearest_point = None
        candidate_point = None
        for i in range(self.num_sides):
            j = (i + 1) % self.num_sides
            p1, p2 = self.vertices[i], self.vertices[j]
            t = np.dot(point - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
            
            if 0 <= t <= 1:
                candidate_point = p1 + t * (p2 - p1)
            
            distance = np.linalg.norm(point - candidate_point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = candidate_point
        return nearest_point
    
    def convex_hull(self):
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.vertices)
        return Polygon(vertices=self.vertices[hull.vertices])

    # def simplify(self, tolerance):
    #     polygon = ShapelyPolygon(self.vertices)
    #     simplified_polygon = polygon.simplify(tolerance)
    #     return Polygon(vertices=np.array(simplified_polygon.exterior.coords[:-1]))

    def __str__(self):
        return f"Polygon(vertices={self.vertices}, num_sides={self.num_sides}, side_length={self.side_length})"