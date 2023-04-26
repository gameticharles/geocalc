# GeoCalc

GeoCalc is a Python library for geometric calculations, focusing on horizontal and vertical curves, angles, triangles, polygons, leveling, and geodesy in transportation engineering and surveying. The library provides various classes and functions for computing and setting out curves, angles, and more.

## Features

- Compute various curve parameters for horizontal curves (Simple, Circular, and Spiral) and Vertical Curves. Calculate the Point of Intersection (PI), Point of Curvature (PC), and Point of Tangency (PT). Calculate chainages and convert between meters and chainage strings. Support for both metric and imperial units
- Angle computations and conversions, including degree-minute-second (DMS) and decimal degrees
- Triangle calculations including area, perimeter, and angles
- Polygon calculations including area and perimeter
- Levelling calculations for elevation and height differences
- Geodesy calculations for distance and bearing

## Installation

You can install GeoCalc using pip:

```bash
pip install geocalc
```

## Usage

# Horizontal Curves

```python
from geocalc import HorizontalCurve, CircularCurves, SpiralCurves

# Create a circular curve

curve = CircularCurve(radius=400, central_angle=24.533333)
starting_chainage = 4545.500
interval = 20
print(curve)

print("External:", curve.external_distance())
print("Middle Ordinate:", curve.middle_ordinate())
print("Long Chord:", curve.long_chord())
```

## Computing Stake Curve by Coordinates

```python
initial_x, initial_y, azimuth = 5723.183, 3728.947, 326.672222

pi,pc,pt, stake_curve_by_coordinates_table = curve.stake_curve_by_coordinates( interval=interval, initial_x=initial_x, initial_y=initial_y, azimuth=azimuth, PI=starting_chainage)

print(pi,pc,pt)
print("\nComputing Stake Curve by Coordinates:\n", stake_curve_by_coordinates_table)
```

## Create a spiral curve

```python
spiral_curve = SpiralCurves(radius=100, degree_of_curve=2)

# Calculate curve parameters

spiral_degree_of_curve = spiral_curve.spiral_degree_of_curve(Ls=20)
```

# Vertical Curves

```python
from geocalc import VerticalCurve

# Create a vertical curve

vertical_curve = VerticalCurve(PVI_elevation=100, grade_in=2, grade_out=-3, length=200)

# Calculate curve parameters

elevation_at_station = vertical_curve.elevation_at_station(50)
```

# Angles

```python
# Example usage

from geocalc import Angle

angle_deg = Angle(45, 'deg')
print("DMS:", angle_deg.dms)
print("Degrees:", angle_deg.deg)
print("Radians:", angle_deg.rad)
print("Grads:", angle_deg.grad)

# An example with DMS data
angle_dms = Angle((30, 45, 0), 'dms')
print("DMS:", angle_dms.dms)
print("Degrees:", angle_dms.deg)
print("Radians:", angle_dms.rad)
print("Grads:", angle_dms.grad)
```

## Example more usage

```python
angle_converter = Angle()

# Degrees to radians
print("Degrees to radians:", angle_converter.degrees_to_radians(45))

# Radians to degrees
print("Radians to degrees:", angle_converter.radians_to_degrees(math.pi/4))

# Degrees to grads
print("Degrees to grads:", angle_converter.degrees_to_grads(45))

# Grads to degrees
print("Grads to degrees:", angle_converter.grads_to_degrees(50))

# Radians to grads
print("Radians to grads:", angle_converter.radians_to_grads(math.pi/4))

# Grads to radians
print("Grads to radians:", angle_converter.grads_to_radians(50))

# Degrees to DMS
print("Degrees to DMS:", angle_converter.degrees_to_dms(45.12345))

# DMS to degrees
print("DMS to degrees:", angle_converter.dms_to_degrees(45, 7, 24.42))
```

# Triangle

## Example usage

```python
from geocalc import Triangle

triangle = Triangle(a=3, b=4, c=5, A=(0, 0), B=(3, 0))
vertex_c1, vertex_c2 = triangle.calculate_other_coordinates()

print("Possible third vertex coordinates:", vertex_c1, vertex_c2)
```

## Example usage

```python
triangle_area = Triangle()

# Heron's formula
print("Heron's formula:", triangle_area.heron_formula(3, 4, 5))

# Trigonometric formula
print("Trigonometric formula:", triangle_area.trigonometric_formula(3, 4, 90))

# Base-height formula
print("Base-height formula:", triangle_area.base_height_formula(3, 4))

# Coordinates formula
print("Coordinates formula:", triangle_area.coordinates_formula(coords=((0, 0), (3, 0), (0, 4))))
```

## Example usage

```python
triangle = Triangle(3, 4, 5, angle_c=90)

# Heron's formula
print("Heron's formula:", triangle.heron())

# Trigonometric formula
print("Trigonometric formula:", triangle.trigonometric())
```

# Polygon

```python
from geocalc import Polygon

#Create a polygon with coordinates
vertices = [(1613.26, 2418.11), (1806.71, 2523.16), (1942.17, 2366.84), (1901.89, 2203.18), (1652.08, 2259.26)]

polygon = Polygon(vertices)
print("Shoelace formula: ", polygon.shoelace())
print("Triangulation: ", polygon.triangulation())
print("Trapezoidal rule: ", polygon.trapezoidal())
print("Monte Carlo method (10,000 points): ", polygon.monte_carlo(num_points=10000))
print("Green's theorem: ", polygon.greens_theorem())
print("\nCentroid: ", polygon.centroid())
print("Moment of inertia: ", polygon.moment_of_inertia())


print("\nAngle Between Side 1 and 2: ", polygon.angle_between_sides(0,1))
print("Length of side 1: ", polygon.side_length_irregular(0))
print("Length of side 2: ", polygon.side_length_irregular(1))

print("\nIs Point(1806.71, 2523.16) inside: ", polygon.is_point_inside_polygon((1806.71, 2400.16)))
print("Is convex: ", polygon.is_convex())
print("\nBounding box: ", polygon.bounding_box())
print("\nOriginal: ", polygon.vertices)

polygon.scale(2.5)
print("\nScaled by 2.5: ", polygon.vertices)

polygon.scale(1/2.5)
print("\nUnScale by 1/2.5: ", polygon.vertices)

polygon.translate(1.0, 1.0)
print("\nTranslated by (1.0, 1.0): ",polygon.vertices )
print("\nNearest point to polygon: ",polygon.nearest_point_on_polygon((1920.17 ,2200.18) ))

print("\nConvexHull: ",polygon.convex_hull())
```

## Create a new regular polygon

```python
polygon = Polygon(num_sides=5, side_length=4)
print("\n\nArea: ", polygon.area_polygon())
print("Perimeter: ", polygon.perimeter())
print("Interior angle: ", polygon.interior_angle())
print("Exterior angle: ", polygon.exterior_angle())

perimeter = polygon.perimeter()
area = polygon.area_polygon()
circumradius = polygon.circumradius()
inradius = polygon.inradius()

# Create polygon and compute the length from other parameters
polygon = Polygon(num_sides=5)

print("Side length from perimeter:", polygon.get_side_length(perimeter=perimeter))
print("Side length from area:", polygon.get_side_length(area=area))
print("Side length from circumradius:", polygon.get_side_length(circumradius=circumradius))
print("Side length from inradius:", polygon.get_side_length(inradius=inradius))
```

# Levelling

```python
from geocalc import Levelling

starting_tbm = 100.000
closing_tbm = 98.050
data =  [
    ('A', 1.751, None, None),  
    ('B', None, 0.540, None),
    ('C', 0.300, None, 2.100),
    ('D', None, 1.100, None),
    ('E', None, 1.260, None),
    ('F', 1.500, None, 2.300),
    ('G', None, None, 1.100)
]

leveling = Levelling(starting_tbm=starting_tbm, closing_tbm=closing_tbm, k=5)

# Add data
for station, bs, is_, fs in data:
    leveling.add_data(station, bs, is_, fs)

# Calculate reduced levels using HPC algorithm
leveling.compute_heights(method="hpc")
print(f"\n\nNumber of instrument station = {leveling.numberSTN}\n")

# Perform arithmetic checks
arithmetic_results = leveling.arithmetic_check()

print("\nArithmetic Checks:")
print(f"Sum of BS = {arithmetic_results['sum_bs']:.3f}")
print(f"Sum of FS = {arithmetic_results['sum_fs']:.3f}")
print(f"First RL = {arithmetic_results['first_rl']:.3f}")
print(f"Last RL = {arithmetic_results['last_rl']:.3f}")
print(f"Sum of BS - Sum of FS = {arithmetic_results['bs_minus_fs']:.4f}")
print(f"Last RL - First RL = {arithmetic_results['last_rl_minus_first_rl']:.4f}")

if arithmetic_results['is_arithmetic_check_passed']:
    print("Arithmetic Checks are OK.")
else:
    diff = arithmetic_results['bs_minus_fs'] - arithmetic_results['last_rl_minus_first_rl']
    print(f"Arithmetic Checks failed with {diff:.4f} differences")

print(f"\nAllowable misclose = {leveling.allowable_misclose():.4f} mm")
print(f"Misclose = {leveling.misclose:.4f} m ({leveling.misclose * 1000:.4f} mm)" if leveling.misclose is not None else None)
print(f"Leveling Status: {'Work is accepted' if leveling.is_work_accepted() else 'Work is not accepted'}.\n")

print(f"Correction = {round(leveling.correction,5) if leveling.correction is not None else None}")
print(f"Correction per station = {round(leveling.adjustment_per_station,5) if leveling.adjustment_per_station is not None else None}\n")

#Print HPC table
print("HPC:",leveling.get_dataFrame())

```

## Calculate reduced levels using Rise & Fall algorithm

```python
leveling.compute_heights(method="rise_fall")
print(leveling.misclose)        # Print the misclose
print(leveling.adjustedRLs)     # Prints the adjusted RLs

#Include the rounding decimal points
print("Rise & Fall:\n",leveling.get_dataFrame(roundDigits=5))

```

# Geodesy

```python
from geocalc import Geodesy

# Calculate distance and bearing between two points
point_a = (12.4924, 41.8902)  # Colosseum, Rome
point_b = (2.2945, 48.8582)   # Eiffel Tower, Paris

distance, bearing = Geodesy.distance_and_bearing(point_a, point_b)
```

```python
geodesy = Geodesy()

# Haversine distance
distance1 = geodesy.haversine_distance(40.689247, -74.044502, 48.858844, 2.294351)
print(f"Haversine distance: {distance1} meters")

# Vincenty distance
distance2 = geodesy.vincenty_distance(40.689247, -74.044502, 48.858844, 2.294351)
print(f"Vincenty distance: {distance2} meters")

# Area of geodesic polygon
polygon_coordinates = [(30, 0), (30, 10), (40, 10), (40, 0)]
area = geodesy.area_of_geodesic_polygon(polygon_coordinates)
print(f"Area of geodesic polygon: {area} square meters")
```

## Contributing

If you'd like to contribute to GeoCalc, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
