import math
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class HorizontalCurve:
    
    def __init__(self, radius:Optional[float]=None, degree_of_curve:Optional[float]=None, length:Optional[float]=None,
                 tangent:Optional[float]=None, chord:Optional[float]=None, central_angle:Optional[float]=None, deflection_angle:Optional[float]=None, intersection_angle:Optional[float]=None):
        self.radius = radius
        self.degree_of_curve = degree_of_curve
        self.length = length
        self.tangent = tangent
        self.chord = chord
        self.central_angle = central_angle
        self.deflection_angle = deflection_angle
        self.intersection_angle = intersection_angle   #Circular
        self._calculate_missing_properties()

    def _calculate_missing_properties(self):
        """
        Degree of curve (D) and radius (R):

            D = 180 * 100 / (π * R)
            R = 180 * 100 / (π * D)
        Length (L) and central angle (Δ):

            L = R * Δ (Δ should be in radians)
            Δ = L / R
            
        Tangent length (T), radius (R), and deflection angle (δ):

            T = R * tan(δ)
            δ = atan(T / R)
            
        Chord length (C), radius (R), and central angle (Δ):

            C = 2 * R * sin(Δ/2)
            Δ = 2 * arcsin(C / (2 * R))
            
        Central angle (Δ) and deflection angle (δ):

            Δ = 2 * δ
            δ = Δ / 2
        """
        while any(attr is None for attr in [self.radius, self.degree_of_curve, self.length, self.tangent, self.chord, self.central_angle, self.deflection_angle, self.intersection_angle]):
            if self.radius is not None and self.degree_of_curve is None:
                self.degree_of_curve = 180 * 100 / (math.pi * self.radius)
            if self.degree_of_curve is not None and self.radius is None:
                self.radius = 180 * 100 / (math.pi * self.degree_of_curve)
            if self.length is not None and self.central_angle is None:
                self.central_angle = self.length / self.radius # type: ignore
            if self.central_angle is not None and self.radius is not None and self.length is None:
                self.length = self.radius * math.radians(self.central_angle)
            if self.tangent is not None and self.chord is None and self.radius is not None and self.central_angle is not None:
                self.chord = 2 * self.radius * math.sin(math.radians(self.central_angle) / 2)
            # if self.chord is not None and self.radius is not None and self.tangent is None and self.central_angle is not None:
            #     self.tangent = self.radius * math.tan(math.radians(self.central_angle) / 2)
            if self.radius is not None and self.tangent is None and self.deflection_angle is not None:
                self.tangent = self.radius * math.tan(math.radians(self.deflection_angle))
            if self.central_angle is not None:
                self.deflection_angle = self.central_angle / 2
            if self.intersection_angle is not None and self.degree_of_curve is not None and self.central_angle is None:
                self.central_angle = (180 / self.degree_of_curve) * self.intersection_angle
            if self.central_angle is not None and self.intersection_angle is None and self.degree_of_curve is not None:
                self.intersection_angle = (self.degree_of_curve / 180) * self.central_angle


    def compute_offsets(self, interval)->list:
        offsets = []
        if self.radius is not None and self.length is not None:
            for dist in range(0, int(self.length) + 1, interval):
                angle = dist / self.radius
                offset = self.radius * (1 - math.cos(angle))
                offsets.append(offset)
        return offsets

    @staticmethod
    def meters_to_feet(meters)->float:
        return meters * 3.28084

    @staticmethod
    def feet_to_meters(feet)->float:
        return feet / 3.28084

    @staticmethod
    def degrees_to_radians(degrees)->float:
        return math.radians(degrees)

    @staticmethod
    def radians_to_degrees(radians)->float:
        return math.degrees(radians)   
    
    @staticmethod
    def _to_dms(degrees)->str:
        d = int(degrees)
        m = int((degrees - d) * 60)
        s = (degrees - d - m/60) * 3600

        if abs(s - 60) < 0.001:
            s = 0
            m +=1
            
        if abs(m - 60) < 0.001:
            d += 1
                
        return f"{d:02}° {m:02}\" {s:05.02f}'"
    
    @staticmethod
    def _to_degree(dms_string)->float:
        dms_split = dms_string.replace('°', ' ').replace('\'', ' ').replace('\"', ' ').split()
        d = float(dms_split[0])
        m = float(dms_split[1])
        s = float(dms_split[2])
        
        degrees = d + m / 60 + s / 3600
        return degrees
    
    
    def __str__(self):
        return f"Radius: {self.radius}\nDegree of Curve: {self.degree_of_curve}\nLength: {self.length}\nTangent: {self.tangent}\nChord: {self.chord}\nCentral Angle: {self.central_angle}\nDeflection Angle: {self.deflection_angle}\nIntersection Angle: {self.intersection_angle}"

class CircularCurve(HorizontalCurve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """
        middle_ordinate (M): M = R - (R * cos(Δ/2))
        length_of_curve (L): L = R * Δ (in radians)
        apex_distance (E): E = R * (1 - cos(Δ/2))
        cord_drop (F): F = (R * sin(Δ/2)) - (R * sin(Δ/4))
        """
        
    def middle_ordinate(self):
        if self.radius is not None and self.central_angle is not None:
            return self.radius * (1 - math.cos(math.radians(self.central_angle) / 2))
        return None
    
    def long_chord(self):
        if self.radius is not None and self.central_angle is not None:
            return 2 * self.radius * math.sin(math.radians(self.central_angle) / 2)
        return None

    def external_distance(self):
        if self.tangent is not None and self.central_angle is not None:
            return self.tangent * (math.tan(math.radians(self.central_angle) / 4))
        return None

    def arc_to_chord_correction(self):
        if self.length is not None and self.central_angle is not None:
            return self.length - self.chord # type: ignore
        return None

    def curve_correction_for_grade(self, grade):
        if self.radius is not None and self.length is not None:
            return (self.length**2 * grade) / (2 * self.radius * 100)
        return None
    
    def cord_drop(self):
        if self.radius is not None and self.central_angle is not None:
            return (self.radius * math.sin(math.radians(self.central_angle) / 2)) - (self.radius * math.sin(math.radians(self.central_angle) / 4))
        return None
    
    @staticmethod
    def _meters_to_chainage(distance):
        distance = round(distance, 3)
        whole_part = int(distance // 1000)
        fractional_part = distance % 1000
        return f"{whole_part} + {fractional_part:.3f}"

    def calculate_next_whole_chainage(self, PC, interval):
        remainder = PC % interval
        if remainder == 0:
            return PC + interval
        else:
            return PC + (interval - remainder) #((starting_chainage // interval)+1) * interval
    
    def generate_chainages(self, interval, PI, PC, PT):
        next_whole_chainage = self.calculate_next_whole_chainage(PC, interval)
        chainages = [PC]
    
        current_chainage = next_whole_chainage
        while current_chainage < PT:
            chainages.append(current_chainage)
            current_chainage += interval

        if PT > chainages[-1]:
            chainages.append(PT)
            
        return chainages
    
    def calculate_PI_PC_PT(self, PI:Optional[float] = None, PC:Optional[float] = None, PT:Optional[float] = None):
        
        while any(par is None for par in [PI, PC, PT]):
            if PI is not None and self.tangent:
                PC = PI - self.tangent
            if PC is not None and self.length:
                PT = PC + self.length
            if PC is not None and self.tangent:
                PI = PC + self.tangent
            
        print(f"\nPI: {self._meters_to_chainage(PI)}\nPC: {self._meters_to_chainage(PC)}\nPT: {self._meters_to_chainage(PT)}")
        
        return PI, PC, PT
    
    def incremental_chord_deflection_angle(self, interval, PI:Optional[float] = None, PC:Optional[float] = None, PT:Optional[float] = None):
        PI, PC, PT = self.calculate_PI_PC_PT(PI = PI, PC=PC, PT=PT)
        chainages = self.generate_chainages(interval,PI,PC,PT)
        data = {
            "Station/Chainage": [],
            "Incremental Chord": [],
            "Deflection increment": [],
            "Deflection Angle": []
        }
        
        arc_length_from_PC = chainages[1] - PC # type: ignore
        arc_length_to_PT = PT - chainages[-2] # type: ignore
        equal_arc_length = 2 * self.radius * math.sin(math.radians((interval / self.length) * self.central_angle)/2) # type: ignore
        k  = self.central_angle / (2 * self.length) # type: ignore
        comp_def_angle_increment = lambda i: 0 if i == 0 else arc_length_to_PT * k if i == 1 else arc_length_from_PC * k if i == len(chainages)-1 else k * interval
        deflect_angle = 0
        for i in range(len(chainages)):
            incremental_chord = 0 if i == 0 else arc_length_from_PC if i == 1 else arc_length_to_PT if i == len(chainages)-1 else equal_arc_length
            deflection_angle_increment = comp_def_angle_increment(i)
            deflect_angle += deflection_angle_increment
            
            data["Station/Chainage"].append(self._meters_to_chainage(chainages[i]))
            data["Incremental Chord"].insert(i, incremental_chord)
            data["Deflection increment"].insert(i,  self._to_dms(deflection_angle_increment))
            data["Deflection Angle"].insert(i,  self._to_dms(deflect_angle))
        
        result = pd.DataFrame(data)
        print("\nIncremental Chord and Deflection Angle:\n", result)
        return result

    def plot_incremental_chord_deflection_angle(self, df):       
        _, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot Incremental Chord
        ax[0].plot(df['Station/Chainage'], df['Incremental Chord'], marker='o')
        ax[0].set_ylabel("Incremental Chord")
        ax[0].grid()

        # Plot Deflection increment and Deflection Angle
        ax[1].plot(df['Station/Chainage'], df['Deflection increment'], marker='o', label='Deflection increment')
        ax[1].plot(df['Station/Chainage'], df['Deflection Angle'], marker='s', label='Deflection Angle')
        ax[1].set_xlabel("Station/Chainage")
        ax[1].set_ylabel("Deflection")
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.show()

    def stakes_curve_by_coordinates(self, interval, initial_x, initial_y, azimuth, PI: Optional[float] = None, PC: Optional[float] = None, PT: Optional[float] = None):
        PI, PC, PT = self.calculate_PI_PC_PT(PI=PI, PC=PC, PT=PT)
        chainages = self.generate_chainages(interval, PI, PC, PT)
        data = {
            "Station/Chainage": [],
            "Station Difference": [],
            "Total Deflection": [],
            "Total Chord": [],
            "Chord Azimuth": [],
            "Departure": [],
            "Latitude": [],
            "X": [],
            "Y": []
        }
        
        # Compute coordinates for PI, PC, and PT
        pi_x, pi_y = self.calculate_point_coordinates(initial_x, initial_y, azimuth, self.tangent)
        pc_x, pc_y = self.calculate_point_coordinates(initial_x, initial_y, azimuth, self.tangent - self.external_distance()) # type: ignore
        pt_x, pt_y = self.calculate_point_coordinates(initial_x, initial_y, azimuth, self.tangent + self.external_distance()) # type: ignore
       
        for chainage in chainages:
            delta_s = chainage - PC # type: ignore
            central_angle_increment = (delta_s / self.length) * self.central_angle
            total_deflection = self.deflection_angle + central_angle_increment / 2
            total_chord = 2 * self.radius * math.sin(math.radians(central_angle_increment) / 2) # type: ignore
            chord_azimuth = azimuth + total_deflection
            departure = total_chord * math.sin(math.radians(chord_azimuth))
            latitude = total_chord * math.cos(math.radians(chord_azimuth))
            x = initial_x + departure
            y = initial_y + latitude

            data["Station/Chainage"].append(self._meters_to_chainage(chainage))
            data["Station Difference"].append(delta_s)
            data["Total Deflection"].append(self._to_dms(total_deflection))
            data["Total Chord"].append(total_chord)
            data["Chord Azimuth"].append(chord_azimuth)
            data["Departure"].append(departure)
            data["Latitude"].append(latitude)
            data["X"].append(x)
            data["Y"].append(y)

        return (pi_x, pi_y), (pc_x, pc_y), (pt_x, pt_y), pd.DataFrame(data)
    
    def stake_curve_by_coordinates(self, interval, initial_x, initial_y, azimuth, PI: Optional[float] = None, PC: Optional[float] = None, PT: Optional[float] = None):
        PI, PC, PT = self.calculate_PI_PC_PT(PI=PI, PC=PC, PT=PT)
        chainages = self.generate_chainages(interval, PI, PC, PT)
        data = {
            "Station/Chainage": [],
            "Station Difference": [],
            "Total Deflection": [],
            "Total Chord": [],
            "Chord Azimuth": [],
            "Departure": [],
            "Latitude": [],
            "X": [],
            "Y": []
        }
        
        # Compute coordinates for PI, PC, and PT
        pi_x, pi_y = initial_x, initial_y,
        pc_x, pc_y = self.calculate_point_coordinates(initial_x, initial_y, azimuth-180, self.tangent) # type: ignore
       
        arc_length_from_PC = chainages[1] - PC # type: ignore
        arc_length_to_PT = PT - chainages[-2] # type: ignore
        equal_arc_length = 2 * self.radius * math.sin(math.radians((interval / self.length) * self.central_angle)/2) # type: ignore
        k  = self.central_angle / (2 * self.length) # type: ignore
        comp_def_angle_increment = lambda i: 0 if i == 0 else arc_length_to_PT * k if i == 1 else arc_length_from_PC * k if i == len(chainages)-1 else k * interval

        total_chord =0
        total_deflection = 0
        i = 0
        for chainage in chainages:

            incremental_chord = 0 if i == 0 else arc_length_from_PC if i == 1 else arc_length_to_PT if i == len(chainages)-1 else equal_arc_length
            deflection_angle_increment = comp_def_angle_increment(i)
            
            total_chord += round(incremental_chord,3)
            total_deflection += deflection_angle_increment            
            chord_azimuth =  0 if i == 0 else azimuth + total_deflection
            departure = total_chord * math.sin(math.radians(chord_azimuth))
            latitude = total_chord * math.cos(math.radians(chord_azimuth))
            x = pc_x + departure
            y = pc_y + latitude

            data["Station/Chainage"].append(self._meters_to_chainage(chainage))
            data["Station Difference"].append(round(chainage - PC,3))# type: ignore
            data["Total Deflection"].append(self._to_dms(total_deflection))
            data["Total Chord"].append(total_chord)
            data["Chord Azimuth"].append(self._to_dms(chord_azimuth))
            data["Departure"].append(departure)
            data["Latitude"].append(latitude)
            data["X"].append(round(x,3))
            data["Y"].append(round(y,3))
            i+=1

        pt_x, pt_y =  data["X"][-1], data["Y"][-1] # type: ignore
        return (pi_x, pi_y), (pc_x, pc_y), (pt_x, pt_y), pd.DataFrame(data)

    def calculate_point_coordinates(self, initial_x, initial_y, azimuth, distance):
        x = initial_x + distance * math.sin(math.radians(azimuth))
        y = initial_y + distance * math.cos(math.radians(azimuth))
        return x, y
        
    def plot_stake_curve_coordinates(self, df, pi, pc, pt):
        if df is None:
            return

        _, ax = plt.subplots(figsize=(8, 8))

        # Plot stake curve coordinates
        ax.plot(df['X'], df['Y'], marker='o', linestyle='-', label='Stake Curve')

        # Add labels for each station/chainage
        for index, row in df.iterrows():
            ax.text(row['X'], row['Y'], f"{row['Station/Chainage']}")

        # Plot PI, PC, and PT stations
        ax.plot(pi[0], pi[1], marker='X', color='red', markersize=10, label='PI')
        ax.text(pi[0], pi[1], 'PI', fontsize=12, color='red', ha='right', va='bottom')
        ax.plot(pc[0], pc[1], marker='X', color='green', markersize=10, label='PC')
        ax.text(pc[0], pc[1], 'PC', fontsize=12, color='green', ha='right', va='bottom')
        ax.plot(pt[0], pt[1], marker='X', color='blue', markersize=10, label='PT')
        ax.text(pt[0], pt[1], 'PT', fontsize=12, color='blue', ha='right', va='bottom')

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Computing Stake Curve by Coordinates")
        ax.legend()
        ax.grid()
        ax.axis('equal')

        plt.tight_layout()
        plt.show()

    def radial_length_azimuth_by_coordinate(self, interval, initial_x, initial_y, azimuth, PI: Optional[float] = None, PC: Optional[float] = None, PT: Optional[float] = None):
        PI, PC, PT = self.calculate_PI_PC_PT(PI=PI, PC=PC, PT=PT)
        chainages = self.generate_chainages(interval, PI, PC, PT)
        
        # Compute coordinates for PI, PC, and PT
        pi_x, pi_y = initial_x, initial_y,
        pc_x, pc_y = self.calculate_point_coordinates(initial_x, initial_y, azimuth-180, self.tangent) # type: ignore
        
        data = {
            "Station/Chainage": [],
            "Departure": [],
            "Latitude": [],
            "X": [],
            "Y": [],
            "Length": [],
            "Azimuth": []
        }

        for chainage in chainages:
            delta_s = chainage - PC # type: ignore
            central_angle_increment = (delta_s / self.length) * self.central_angle
            radial_length = self.radius / math.cos(math.radians(central_angle_increment) / 2) # type: ignore
            radial_azimuth = azimuth + self.deflection_angle + central_angle_increment / 2
            departure = radial_length * math.sin(math.radians(radial_azimuth))
            latitude = radial_length * math.cos(math.radians(radial_azimuth))
            x = pc_x + departure
            y = pc_y + latitude

            data["Station/Chainage"].append(self._meters_to_chainage(chainage))
            data["Departure"].append(departure)
            data["Latitude"].append(latitude)
            data["X"].append(x)
            data["Y"].append(y)
            data["Length"].append(radial_length)
            data["Azimuth"].append(radial_azimuth)

        return pd.DataFrame(data)

    def tangent_offset(self, starting_chainage, interval):
        PI, PC, PT, chainages = self.calculate_PI_PC_PT(starting_chainage, interval) # type: ignore
        data = {
            "Station/Chainage": [],
            "Deflection Angle": [],
            "Chord": [],
            "Tangent_Offset": [],
            "Tangent_Distance": []
        }

        for chainage in chainages:
            delta_s = chainage - PC
            central_angle_increment = (delta_s / self.length) * self.central_angle
            deflection_angle_increment = central_angle_increment / 2
            chord = 2 * self.radius * math.sin(math.radians(central_angle_increment) / 2) # type: ignore
            tangent_offset = self.radius * (1 - math.cos(math.radians(central_angle_increment) / 2)) # type: ignore
            tangent_distance = self.radius * math.tan(math.radians(deflection_angle_increment))# type: ignore

            data["Station/Chainage"].append(chainage)
            data["Deflection Angle"].append(self.deflection_angle + deflection_angle_increment)
            data["Chord"].append(chord)
            data["Tangent_Offset"].append(tangent_offset)
            data["Tangent_Distance"].append(tangent_distance)

        return pd.DataFrame(data)

class SpiralCurve(HorizontalCurve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_of_transition = None
        self.rate_of_change_of_curvature = None

        if self.radius and self.degree_of_curve:
            self.length_of_transition = self.radius * self.degree_of_curve
            self.rate_of_change_of_curvature = self.degree_of_curve / self.length_of_transition
            
        """
        spiral_degree_of_curve: Ds = D0 + (Ls * R0 / LT)
        spiral_tangent_length: Ts = LT / 2 * (1 - cos(Δ/2))
        spiral_chord_length: Cs = 2 * LT * sin(Δ/2)
        spiral_offset: Os = LT * (1 - cos(Δ/2))
        spiral_radius: Rs = R0 + (Ls / LT) * (R - R0)
        """

    def degree_of_curve_spiral(self):
        if self.radius is not None and self.length is not None:
            return 1 / (self.length / (2 * self.radius * math.pi))
        return None
    
    def spiral_degree_of_curve(self, Ls):
        if self.degree_of_curve is not None and self.radius is not None and self.length_of_transition is not None:
            return self.degree_of_curve + (Ls * self.radius / self.length_of_transition)
        return None

    def spiral_tangent_length(self, Ls):
        if self.length_of_transition is not None and self.central_angle is not None:
            return (self.length_of_transition / 2) * (1 - math.cos(math.radians(self.central_angle) / 2))
        return None

    def spiral_chord_length(self, Ls):
        if self.length_of_transition is not None and self.central_angle is not None:
            return 2 * self.length_of_transition * math.sin(math.radians(self.central_angle) / 2)
        return None

    def spiral_offset(self, Ls):
        if self.length_of_transition is not None and self.central_angle is not None:
            return self.length_of_transition * (1 - math.cos(math.radians(self.central_angle) / 2))
        return None

    def spiral_radius(self, Ls):
        if self.radius is not None and self.length_of_transition is not None:
            return self.radius + (Ls / self.length_of_transition) * (self.radius - self.radius)
        return None

class VerticalCurve:
    def __init__(self, radius:Optional[float]=None, degree_of_curve:Optional[float]=None, length:Optional[float]=None,
                 tangent:Optional[float]=None, chord:Optional[float]=None, central_angle:Optional[float]=None, deflection_angle:Optional[float]=None, intersection_angle:Optional[float]=None):
        self.radius = radius
        self.degree_of_curve = degree_of_curve
        self.length = length
        self.tangent = tangent
        self.chord = chord
        self.central_angle = central_angle
        self.deflection_angle = deflection_angle
        self.intersection_angle = intersection_angle   #Circular
        self._calculate_missing_properties()

    def _calculate_missing_properties(self):
        """
        Degree of curve (D) and radius (R):

            D = 180 * 100 / (π * R)
            R = 180 * 100 / (π * D)
        Length (L) and central angle (Δ):

            L = R * Δ (Δ should be in radians)
            Δ = L / R
            
        Tangent length (T), radius (R), and deflection angle (δ):

            T = R * tan(δ)
            δ = atan(T / R)
            
        Chord length (C), radius (R), and central angle (Δ):

            C = 2 * R * sin(Δ/2)
            Δ = 2 * arcsin(C / (2 * R))
            
        Central angle (Δ) and deflection angle (δ):

            Δ = 2 * δ
            δ = Δ / 2
        """
        while any(attr is None for attr in [self.radius, self.degree_of_curve, self.length, self.tangent, self.chord, self.central_angle, self.deflection_angle, self.intersection_angle]):
            if self.radius is not None and self.degree_of_curve is None:
                self.degree_of_curve = 180 * 100 / (math.pi * self.radius)
            if self.degree_of_curve is not None and self.radius is None:
                self.radius = 180 * 100 / (math.pi * self.degree_of_curve)
            if self.length is not None and self.central_angle is None:
                self.central_angle = self.length / self.radius # type: ignore
            if self.central_angle is not None and self.radius is not None and self.length is None:
                self.length = self.radius * math.radians(self.central_angle)
            if self.tangent is not None and self.chord is None and self.radius is not None and self.central_angle is not None:
                self.chord = 2 * self.radius * math.sin(math.radians(self.central_angle) / 2)
            # if self.chord is not None and self.radius is not None and self.tangent is None and self.central_angle is not None:
            #     self.tangent = self.radius * math.tan(math.radians(self.central_angle) / 2)
            if self.radius is not None and self.tangent is None and self.deflection_angle is not None:
                self.tangent = self.radius * math.tan(math.radians(self.deflection_angle))
            if self.central_angle is not None:
                self.deflection_angle = self.central_angle / 2
            if self.intersection_angle is not None and self.degree_of_curve is not None and self.central_angle is None:
                self.central_angle = (180 / self.degree_of_curve) * self.intersection_angle
            if self.central_angle is not None and self.intersection_angle is None and self.degree_of_curve is not None:
                self.intersection_angle = (self.degree_of_curve / 180) * self.central_angle
