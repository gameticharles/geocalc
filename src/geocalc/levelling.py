import os
import math
import pandas as pd
from typing import Optional, List

class Levelling:
    def __init__(self, starting_tbm, closing_tbm=None, k=3, roundDigits = 4, file_path=None):
        """
    Initialize the Levelling class with the given parameters.

    :param starting_tbm: The initial reduced level (TBM) at the start of the levelling process.
    :type starting_tbm: float
    :param closing_tbm: The final reduced level (TBM) at the end of the levelling process (Optional).
    :type closing_tbm: Optional[float]
    :param k: The constant value used to compute the allowable misclose. (Optional, default is 3)
    :type k: int
    :param roundDigits: The number of decimal places to round the results. (Optional, default is 4)
    :type roundDigits: int
    :param file_path: The path to the data file containing the leveling data (Optional).
    :type file_path: Optional[str]

    :ivar starting_tbm: The initial reduced level (TBM) at the start of the levelling process.
    :ivar closing_tbm: The final reduced level (TBM) at the end of the levelling process.
    :ivar k: The constant value used to compute the allowable misclose.
    :ivar data: A list containing the leveling data as dictionaries.
    :ivar roundDigits: The number of decimal places to round the results.
    :ivar results: A pandas DataFrame containing the computed results.
    :ivar numberSTN: The number of instrument stations in the leveling process.
    :ivar misclose: The computed misclose value.
    :ivar correction: The computed correction value.
    :ivar adjustment_per_station: The computed adjustment per station value.
    :ivar reducedLevels: A list of computed reduced levels.
    :ivar adjustedRLs: A list of adjusted reduced levels.
    :ivar adjustments: A list of computed adjustments.
    :ivar method: The method used to compute the heights ("rise_fall" or "hpc").
    """
        self.starting_tbm = starting_tbm
        self.closing_tbm = closing_tbm
        self.k = k
        self.data = []
        self.roundDigits = roundDigits
        self.results = pd.DataFrame()
        self.numberSTN:int = 0
        self.misclose:Optional[float] = None
        self.correction:Optional[float] = None
        self.adjustment_per_station:Optional[float] = None
        self.reducedLevels:list[float] = []
        self.adjustedRLs:list[float] = []
        self.adjustments:list[float] = []
        self.method = "rise_fall"
        if file_path:
            self.read_from_file(file_path)

    def read_from_file(self, file_path):
        """
        Read leveling data from a file and add it to the data attribute.

        The file should contain one line per observation with the following format:
        <station> <BS> <IS> <FS>
        where <station> is the station name, and <BS>, <IS>, <FS> are the back sight,
        intermediate sight, and fore sight readings, respectively.

        Parameters:
        -----------
        file_path : str
            The path to the data file containing the leveling data.

        Raises:
        -------
        FileNotFoundError
            If the specified file is not found.

        Example:
        --------
        >>> levelling = Levelling(100.0)
        >>> levelling.read_from_file("data.txt")
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File not found.")

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    self.add_data(parts[0], float(parts[1]), float(parts[2]),float(parts[3]))
                    

    def add_data(self, station:str, bs:float, is_:float, fs:float):
        """
        Add a single leveling observation to the data attribute.

        Parameters:
        -----------
        station : str
            The name of the leveling station.
        bs : float
            The back sight reading at the station.
        is_ : float
            The intermediate sight reading at the station (if any).
        fs : float
            The fore sight reading at the station.

        Example:
        --------
        >>> levelling = Levelling(100.0)
        >>> levelling.add_data("A", 1.5, None, 2.3)
        """
        self.data.append({'Remarks': station, 'BS': bs, 'IS': is_, 'FS': fs})

    def compute_misclose(self):
        """
        Compute the misclose between the last reduced level and the closing TBM.
        Sets the misclose attribute of the Levelling object. If closing_tbm is not provided,
        the function will return None and the misclose attribute will not be set.

        Example:
        --------
        >>> levelling = Levelling(100.0, closing_tbm=120.0)
        >>> levelling.compute_misclose()
        >>> print(f"Misclose: {levelling.misclose}")
        """
        if self.closing_tbm is None:
            return None
        self.misclose = self.reducedLevels[-1] - self.closing_tbm

    def allowable_misclose(self):
        """
        Calculate the allowable misclose based on the number of instrument stations.
        The allowable misclose is computed as k * sqrt(number of stations), where k is a constant.

        Returns:
        --------
        float
            The allowable misclose value.

        Example:
        --------
        >>> levelling = Levelling(100.0)
        >>> levelling.numberSTN = 5
        >>> print(f"Allowable misclose: {levelling.allowable_misclose()}")
        """
        return self.k * (self.numberSTN ** 0.5)

    def is_work_accepted(self):
        """
        Determine if the leveling work is accepted based on the misclose value and the allowable misclose.
        The work is accepted if the absolute value of misclose (in millimeters) is less than or equal
        to the allowable misclose.

        Returns:
        --------
        bool or None
            True if the work is accepted, False if not accepted, and None if misclose is not available.

        Example:
        --------
        >>> levelling = Levelling(100.0, 103.0)
        >>> levelling.numberSTN = 5
        >>> levelling.misclose = 0.002
        >>> print(f"Work accepted: {levelling.is_work_accepted()}")
        """ 
        if self.misclose is None:
            return
        
        if abs(self.misclose*1000) <= self.allowable_misclose():
            return True
        else:
            return False
        
    def arithmetic_check(self):
        """
        Perform arithmetic checks for the leveling data. This function checks if the sum of BS
        minus the sum of FS is equal to the difference between the last and the first reduced levels.

        Returns:
        --------
        dict
            A dictionary containing the following key-value pairs:
                - 'sum_bs': Sum of back sights
                - 'sum_fs': Sum of fore sights
                - 'bs_minus_fs': Difference between sum of BS and sum of FS
                - 'first_rl': First reduced level
                - 'last_rl': Last reduced level
                - 'last_rl_minus_first_rl': Difference between the last and the first reduced levels
                - 'is_arithmetic_check_passed': True if arithmetic check passes, otherwise False
                - 'allowable_misclose': Allowable misclose
                - 'misclose': Misclose value, None if not available
                - 'work_status': True if work is accepted, False if not accepted, None if misclose is not available

        Example:
        --------
        >>> levelling = Levelling(100.0, 103.0)
        >>> levelling.add_data("A", 1.0, None, None)
        >>> levelling.add_data("B", None, None, 0.5)
        >>> levelling.compute_heights()
        >>> arithmetic_check_results = levelling.arithmetic_check()
        >>> print(f"Arithmetic check passed: {arithmetic_check_results['is_arithmetic_check_passed']}")
        """
        bs_sum = round(sum(d['BS'] for d in self.data if d['BS'] is not None),self.roundDigits)
        fs_sum = round(sum(d['FS'] for d in self.data if d['FS'] is not None),self.roundDigits)
        LastRL_FirstRL = self.reducedLevels[-1] - self.reducedLevels[0]
        
        return {
            "sum_bs": bs_sum,
            "sum_fs": fs_sum,
            "bs_minus_fs": round(bs_sum - fs_sum, self.roundDigits),
            "first_rl":self.reducedLevels[0],
            "last_rl":self.reducedLevels[-1],
            "last_rl_minus_first_rl": round(LastRL_FirstRL, self.roundDigits),
            "is_arithmetic_check_passed": math.isclose(bs_sum - fs_sum, LastRL_FirstRL, rel_tol=1e-6),
            "allowable_misclose": round(self.allowable_misclose(), self.roundDigits),
            "misclose": round(self.misclose, self.roundDigits) if self.misclose is not None else None,
            "work_status": self.is_work_accepted()
        }
        
    def adjust_heights(self):
        """
        Adjust the reduced levels based on the calculated misclose value. The misclose is
        distributed evenly across all the leveling stations. The adjusted reduced levels
        are stored in 'self.adjustedRLs', and the adjustments applied to each station are
        stored in 'self.adjustments'.

        Note: This function should be called only after 'compute_misclose' has been called.

        Example:
        --------
        >>> levelling = Levelling(100.0, 103.0)
        >>> levelling.add_data("A", 1.0, None, None)
        >>> levelling.add_data("B", None, None, 0.5)
        >>> levelling.compute_heights()
        >>> levelling.compute_misclose()
        >>> levelling.adjust_heights()
        >>> print(f"Adjusted reduced levels: {levelling.adjustedRLs}")
        """
        if self.misclose is None:
            return
        
        self.correction = -1 * self.misclose
        self.adjustment_per_station = self.correction / self.numberSTN
        
        countBS = 0
        self.adjustedRLs = self.reducedLevels.copy()
        self.adjustments = [0.0]
        adjustment = 0.0
        for i, d in enumerate(self.data):
            if d['BS'] is not None:
                countBS += 1
                adjustment = countBS * self.adjustment_per_station
            
            if i != 0:
                self.adjustments.append(adjustment)
                self.adjustedRLs[i] = self.reducedLevels[i] + self.adjustments[i]

    def compute_heights(self, method="rise_fall"):
        """
        Computes the reduced levels of the levelling data using the specified method
        ('rise_fall' or 'hpc'), then computes the misclose and adjusts the reduced levels
        if necessary.

        Parameter:
        -----------
        method : str, optional
            The method to use for computing the reduced levels. Options are 'rise_fall'
            (default) or 'hpc'.
            
        Returns:
        --------
        list
            A list containing the original reduced levels, the adjustments, and the
            adjusted reduced levels. Otherwise, only the original reduced levels are returned.

        Raises:
        -------
        ValueError
            If an invalid method is provided or if there is insufficient data for the computation.

        Example:
        --------
        >>> levelling = Levelling(100.0, 103.0)
        >>> levelling.add_data("A", 1.0, None, None)
        >>> levelling.add_data("B", None, None, 0.5)
        >>> results = levelling.compute_heights()
        >>> print(f"Reduced levels: {results[0]}")
        >>> if len(results) > 1:
        >>>     print(f"Adjustments: {results[1]}")
        >>>     print(f"Adjusted reduced levels: {results[2]}")
        """
        if method not in ["rise_fall", "hpc"]:
            raise ValueError("Invalid method. Allowed methods are 'rise_fall' and 'hpc'.")

        if not self.data:
            raise ValueError("No data provided.")
        
        #Compute the number of stations
        self.numberSTN = sum(1 for d in self.data if d['BS'] is not None)
        
        self.reducedLevels = [self.starting_tbm]
        self.method = method

        if method == "rise_fall":
            self.rise:List[Optional[float]] = [None]
            self.fall:List[Optional[float]] = [None]
        
            for i in range(len(self.data) - 1):
                curr_data = self.data[i]
                next_data = self.data[i + 1]
                rise_fall = 0.0
                
                if curr_data['BS'] is not None and next_data['FS'] is not None:
                    rise_fall = curr_data['BS'] - next_data['FS']
                elif curr_data['BS'] is not None and next_data['IS'] is not None:
                    rise_fall = curr_data['BS'] - next_data['IS']
                elif curr_data['IS'] is not None and next_data['IS'] is not None:
                    rise_fall = curr_data['IS'] - next_data['IS']
                elif curr_data['IS'] is not None and next_data['FS'] is not None:
                    rise_fall = curr_data['IS'] - next_data['FS']
                else:
                    raise ValueError("Insufficient data for Rise & Fall computation.")

                if rise_fall > 0:
                    self.rise.append(rise_fall)
                    self.fall.append(None) 
                else:
                    self.fall.append(rise_fall) 
                    self.rise.append(None)
                    
                self.reducedLevels.append(self.reducedLevels[-1] + rise_fall)
                
        else:  # method == "hpc"
            hpc = 0
            self.hpc= [None] * len(self.data)
            for i, d in enumerate(self.data):
                if d['BS'] is not None and d['FS'] is None:
                    hpc = self.reducedLevels[i] + d['BS']
                    self.hpc[i] = hpc
                    continue
                elif d['BS'] is not None and d['FS'] is not None:   
                    self.reducedLevels.append(hpc - d['FS'])
                    hpc = self.reducedLevels[i] + d['BS']
                    self.hpc[i] = hpc
                    continue    
                
                self.reducedLevels.append(hpc - d['FS'] if d['FS'] is not None else hpc - d['IS'])
        
        #Compute the misclose
        self.compute_misclose()
        
        #Adjust the Reduce Levels if misclose is not None
        self.adjust_heights()
        
        return [self.reducedLevels, self.adjustments, self.adjustedRLs] if self.misclose is not None else self.reducedLevels

    def get_dataFrame(self, roundDigits = 4):
        """
        Creates and returns a Pandas DataFrame containing the levelling data, computed
        reduced levels, and, if applicable, the adjustments and adjusted reduced levels.

        Parameters:
        -----------
        roundDigits : int, optional
            The number of decimal places to round the values in the DataFrame (default is 4).

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the levelling data, computed reduced levels, and
            adjustments and adjusted reduced levels if a misclose has been detected.

        Raises:
        -------
        ValueError
            If compute_heights() has not been called before getting the DataFrame.

        Example:
        --------
        >>> levelling = Levelling(100.0, 103.0)
        >>> levelling.add_data("A", 1.0, None, None)
        >>> levelling.add_data("B", None, None, 0.5)
        >>> levelling.compute_heights()
        >>> df = levelling.get_dataFrame()
        >>> print(df)
        """
        if not hasattr(self, 'reducedLevels'):
            raise ValueError("compute_heights() must be called before getting the DataFrame.")

        self.roundDigits = roundDigits
        
        #Compute the misclose
        self.compute_misclose()
        
        #Adjust the Reduce Levels if misclose is not None
        self.adjust_heights()
        
        if self.method == "rise_fall":
            df = pd.DataFrame(self.data)
            df['Rise'] = self.rise
            df['Fall'] = self.fall
            df['Reduced Level (RL)'] = self.reducedLevels
            if self.misclose is not None:
                df['Adjustment'] = self.adjustments
                df['Adjusted RL'] = self.adjustedRLs
        else:  # method == "hpc"
            df = pd.DataFrame(self.data)
            df['HPC'] = self.hpc
            df['Reduced Level (RL)'] = self.reducedLevels
            if self.misclose is not None:
                df['Adjustment'] = self.adjustments
                df['Adjusted RL'] = self.adjustedRLs
        
        #MFormat to the required decimal places
        df = df.round(self.roundDigits)
        return df.fillna("")
    
    def __repr__(self):
        return str(self.get_dataFrame())