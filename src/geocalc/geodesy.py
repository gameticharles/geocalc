import math

class Geodesy:
    def __init__(self, a=6378137, f=1/298.257223563):
        self.a = a  # Semi-major axis of the reference ellipsoid (WGS 84 by default)
        self.f = f  # Flattening of the reference ellipsoid (WGS 84 by default)
        self.b = self.a * (1 - f)  # Semi-minor axis

    @staticmethod
    def to_radians(degrees):
        return math.radians(degrees)

    @staticmethod
    def to_degrees(radians):
        return math.degrees(radians)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates the great-circle distance between two points on a sphere."""
        lat1, lon1, lat2, lon2 = map(self.to_radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return self.a * c

    def vincenty_distance(self, lat1, lon1, lat2, lon2, max_iter=200, tolerance=1e-12):
        """Calculates the distance between two points on an ellipsoid using Vincenty's formula."""
        lat1, lon1, lat2, lon2 = map(self.to_radians, [lat1, lon1, lat2, lon2])

        U1 = math.atan((1 - self.f) * math.tan(lat1))
        U2 = math.atan((1 - self.f) * math.tan(lat2))
        L = lon2 - lon1
        lambda_ = L

        for _ in range(max_iter):
            sin_lambda = math.sin(lambda_)
            cos_lambda = math.cos(lambda_)
            sin_sigma = math.sqrt((math.cos(U2) * sin_lambda) ** 2 + (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * cos_lambda) ** 2)
            cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * cos_lambda
            sigma = math.atan2(sin_sigma, cos_sigma)

            sin_alpha = math.cos(U1) * math.cos(U2) * sin_lambda / sin_sigma
            cos_sq_alpha = 1 - sin_alpha ** 2
            cos_2_sigma_m = cos_sigma - 2 * math.sin(U1) * math.sin(U2) / cos_sq_alpha

            C = self.f / 16 * cos_sq_alpha * (4 + self.f * (4 - 3 * cos_sq_alpha))
            lambda_prev = lambda_
            lambda_ = L + (1 - C) * self.f * sin_alpha * (sigma + C * sin_sigma * (cos_2_sigma_m + C * cos_sigma * (-1 + 2 * cos_2_sigma_m ** 2)))

            if abs(lambda_ - lambda_prev) < tolerance:
                break
        else:
            # If Vincenty's formula didn't converge, fallback to haversine formula
            return self.haversine_distance(math.degrees(lat1), math.degrees(lon1), math.degrees(lat2), math.degrees(lon2))

        u_sq = cos_sq_alpha * (self.a**2 - self.b**2) / (self.b**2)
        A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
        delta_sigma = B * sin_sigma * (cos_2_sigma_m + (B / 4) * (cos_sigma * (-1 + 2 * cos_2_sigma_m**2) - (B / 6) * cos_2_sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos_2_sigma_m**2)))

        s = self.b * A * (sigma - delta_sigma)

        return s

    def area_of_geodesic_polygon(self, coordinates):
        """Calculates the area of a geodesic polygon using the method of spherical excess."""
        EARTH_RADIUS = 6371009  # Mean Earth radius in meters

        coordinates = [(self.to_radians(lat), self.to_radians(lon)) for lat, lon in coordinates]
        coordinates.append(coordinates[0])  # Close the polygon

        total_angle = 0
        for i in range(len(coordinates) - 1):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[i + 1]
            total_angle += (lon2 - lon1) * (2 + math.sin(lat1) * math.sin(lat2))

        # Calculate the spherical excess and multiply by the Earth's radius squared to get the area
        spherical_excess = total_angle - (len(coordinates) - 3) * math.pi
        area = abs(spherical_excess) * EARTH_RADIUS**2

        return area
