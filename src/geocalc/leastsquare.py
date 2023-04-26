import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import Ridge, Lasso, ElasticNet

class LeastSquaresAdjustment:
    """
    A class for performing weighted least squares adjustment on a set of observations.
    
    Attributes:
        observations (numpy.ndarray): The observations vector.
        design_matrix (numpy.ndarray): The design matrix.
        weight_matrix (numpy.ndarray): The weight matrix.
    """

    def __init__(self, observations, design_matrix, weight_matrix=None):
        """
        Initializes the LeastSquaresAdjustment class with observations, design matrix, and an optional weight matrix.
        
        Args:
            observations (list): The observations vector.
            design_matrix (list): The design matrix.
            weight_matrix (list, optional): The weight matrix. Defaults to an identity matrix.
        """
        self.validate_input(observations, design_matrix, weight_matrix)
        self.observations = np.array(observations)
        self.design_matrix = np.array(design_matrix)

        if weight_matrix is None:
            self.weight_matrix = np.identity(len(observations))
        else:
            self.weight_matrix = np.array(weight_matrix)

    @staticmethod
    def validate_input(observations, design_matrix, weight_matrix):
        """
        Validates the input matrices for compatibility.
        
        Args:
            observations (list): The observations vector.
            design_matrix (list): The design matrix.
            weight_matrix (list, optional): The weight matrix.
        """
        if len(observations) != len(design_matrix):
            raise ValueError("The number of rows in the observations and design matrix must be the same.")

        if weight_matrix is not None and len(observations) != len(weight_matrix):
            raise ValueError("The number of rows in the observations and weight matrix must be the same.")

    def update_data(self, observations=None, design_matrix=None, weight_matrix=None):
        """
        Updates the observations, design_matrix, and/or weight_matrix.
        
        Args:
            observations (list, optional): The new observations vector.
            design_matrix (list, optional): The new design matrix.
            weight_matrix (list, optional): The new weight matrix.
        """
        if observations is not None:
            self.validate_input(observations, design_matrix or self.design_matrix, weight_matrix or self.weight_matrix)
            self.observations = np.array(observations)

        if design_matrix is not None:
            self.validate_input(self.observations, design_matrix, weight_matrix or self.weight_matrix)
            self.design_matrix = np.array(design_matrix)

        if weight_matrix is not None:
            self.validate_input(self.observations, self.design_matrix, weight_matrix)
            self.weight_matrix = np.array(weight_matrix)

    def compute_adjustment(self):
        """
        Computes the least squares adjustment for the given observations, design matrix, and weight matrix.
        
        Returns:
            dict: A dictionary containing the estimated parameters, residuals, variance factor, covariance matrix, and adjusted observations.
        """
        # Compute the normal matrix
        normal_matrix = self.design_matrix.T @ self.weight_matrix @ self.design_matrix

        # Compute the right-hand side (rhs) vector
        rhs = self.design_matrix.T @ self.weight_matrix @ self.observations

        # Solve the normal equations
        try:
            parameters = np.linalg.solve(normal_matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise ValueError("Failed to solve the normal equations: " + str(e))

        # Compute residuals
        residuals = self.observations - self.design_matrix @ parameters

        # Compute variance factor
        variance_factor = residuals.T @ self.weight_matrix @ residuals / (len(self.observations) - len(parameters))
        
        # Compute the covariance matrix of the estimated parameters
        covariance_matrix = np.linalg.inv(normal_matrix) * variance_factor

        # Compute adjusted observations
        adjusted_observations = self.design_matrix @ parameters

        return {
            'parameters': parameters,
            'residuals': residuals,
            'variance_factor': variance_factor,
            'covariance_matrix':covariance_matrix,
            'adjusted_observations': adjusted_observations
        }

    def error_ellipse(self):
        """
        Computes the error ellipse for 2D coordinates (X, Y) based on the covariance matrix of the estimated parameters.

        Returns:
            tuple: A tuple containing the semi-major axis (a), semi-minor axis (b), and the orientation angle (angle) in degrees.
        """
        adjustment_results = self.compute_adjustment()
        covariance_matrix = adjustment_results['covariance_matrix']

        if covariance_matrix.shape != (2, 2):
            raise ValueError("The error ellipse can only be computed for 2D coordinates (X, Y).")

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Calculate the semi-major and semi-minor axes of the error ellipse
        a = math.sqrt(eigenvalues[0])
        b = math.sqrt(eigenvalues[1])

        # Calculate the orientation angle of the error ellipse
        angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        return a, b, angle

    def plot_error_ellipse(self, center, a, b, angle, original_points=None):
        """
        Plots the error ellipse given the center, semi-major axis, semi-minor axis, and orientation angle.

        Args:
            center (tuple): A tuple containing the X, Y coordinates of the ellipse's center.
            a (float): The semi-major axis of the error ellipse.
            b (float): The semi-minor axis of the error ellipse.
            angle (float): The orientation angle of the error ellipse in degrees.
            original_points (list of tuples, optional): A list of tuples containing the X, Y coordinates of the original data points.
        """

        # Create a new figure and axis
        _, ax = plt.subplots()

        # Create an ellipse patch with the given parameters
        ellipse = Ellipse(center, 2 * a, 2 * b, angle, edgecolor='r', facecolor='none')

        # Add the ellipse patch to the axis
        ax.add_patch(ellipse)

        # Plot the adjusted point
        ax.plot(center[0], center[1], 'bo', label='Adjusted point')

        # Plot the original data points if provided
        if original_points:
            x_coords, y_coords = zip(*original_points)
            ax.plot(x_coords, y_coords, 'ro', label='Original points')

        # Set the aspect ratio of the plot to 'equal' and add gridlines
        ax.set_aspect('equal', 'box')
        ax.grid(True)

        # Set axis labels and show the legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        # Display the plot
        plt.show()

    def t_score(self, confidence_level, dof):
            # Calculate the t-score using the inverse of the Student's t-distribution
            return math.sqrt(2 / (dof - 2)) * math.gamma((dof - 1) / 2) / math.gamma(dof / 2) * ((1 - confidence_level) / 2)

    def confidence_intervals(self, confidence_level=0.95):
        """
        Computes the confidence intervals for the estimated parameters based on their covariance matrix and a specified confidence level.

        Args:
            confidence_level (float, optional): The desired confidence level (0 < confidence_level < 1). Defaults to 0.95.

        Returns:
            list: A list of tuples containing the lower and upper bounds of the confidence intervals for each parameter.
        """
        
        adjustment_results = self.compute_adjustment()
        covariance_matrix = adjustment_results['covariance_matrix']
        parameters = adjustment_results['parameters']
        dof = len(self.observations) - len(parameters)
        t = self.t_score(confidence_level, dof)

        intervals = []
        for i in range(len(parameters)):
            std_error = np.sqrt(covariance_matrix[i, i])
            margin_of_error = t * std_error
            intervals.append((parameters[i] - margin_of_error, parameters[i] + margin_of_error))

        return intervals
    
    def outlier_detection(self, threshold):
        """
        Identifies outliers in the observations based on their residuals and a specified threshold.

        Args:
            threshold (float): The threshold for identifying outliers.

        Returns:
            list: A list of indices of the identified outlier observations.
        """
        adjustment_results = self.compute_adjustment()
        residuals = adjustment_results['residuals']

        return [i for i, r in enumerate(residuals) if abs(r) > threshold]
    
    def r_squared(self):
        """
        Computes the coefficient of determination (R-squared) for the least squares adjustment.
        Goodness-of-fit measures

        Returns:
            float: The R-squared value.
        """
        adjustment_results = self.compute_adjustment()
        residuals = adjustment_results['residuals']
        total_sum_of_squares = np.sum((self.observations - np.mean(self.observations)) ** 2)
        residual_sum_of_squares = np.sum(residuals ** 2)

        return 1 - (residual_sum_of_squares / total_sum_of_squares)

    def hypothesis_testing(self, null_hypothesis, alpha=0.05):
        """
        Performs a hypothesis test for the estimated parameters to determine if they are significantly different
        from the specified null hypothesis values.

        Args:
            null_hypothesis (list or numpy.ndarray): The null hypothesis values for the parameters.
            alpha (float, optional): The significance level for the hypothesis test. Defaults to 0.05.

        Returns:
            tuple: A tuple containing:
                - reject_null (numpy.ndarray): A boolean array indicating whether to reject the null hypothesis for each parameter.
                - test_statistic (numpy.ndarray): The test statistic values for each parameter.
        """
        adjustment_results = self.compute_adjustment()
        parameters = adjustment_results['parameters']
        covariance_matrix = adjustment_results['covariance_matrix']
        dof = len(self.observations) - len(parameters)

        t_critical = self.t_score(1 - alpha / 2, dof)
        test_statistic = (parameters - null_hypothesis) / np.sqrt(np.diag(covariance_matrix))

        reject_null = np.abs(test_statistic) > t_critical
        return reject_null, test_statistic
    
    def influence_analysis(self, method='cooks_distance'):
        """
        Computes the influence of individual observations on the estimated parameters using different methods.
        
        Args:
            method (str, optional): The method used for influence analysis. Choose from 'basic', 'leverage', or 'cooks_distance'. Defaults to 'cooks_distance'.

        Returns:
            numpy.ndarray: The influence values based on the selected method.

        Raises:
            ValueError: If an invalid method is provided.
        """
        if method not in ['basic', 'leverage', 'cooks_distance']:
            raise ValueError("Invalid method. Choose from 'basic', 'leverage', or 'cooks_distance'.")

        adjustment_results = self.compute_adjustment()
        parameters = adjustment_results['parameters']
        residuals = adjustment_results['residuals']
        h_matrix = self.design_matrix @ np.linalg.inv(self.design_matrix.T @ self.weight_matrix @ self.design_matrix) @ self.design_matrix.T @ self.weight_matrix

        if method == 'basic':
            influences = []
            for i in range(len(self.observations)):
                obs_wo_i = np.delete(self.observations, i)
                design_wo_i = np.delete(self.design_matrix, i, axis=0)
                weight_wo_i = np.delete(self.weight_matrix, i, axis=0)
                weight_wo_i = np.delete(weight_wo_i, i, axis=1)

                lsa_wo_i = LeastSquaresAdjustment(obs_wo_i, design_wo_i, weight_wo_i)
                params_wo_i = lsa_wo_i.compute_adjustment()['parameters']

                influence = np.abs(parameters - params_wo_i)
                influences.append(influence)

            return influences

        elif method == 'leverage':
            leverage = np.diag(h_matrix)
            return leverage

        elif method == 'cooks_distance':
            leverage = np.diag(h_matrix)
            cooks_distance = (residuals**2 * leverage) / (len(parameters) * (1 - leverage))
            return cooks_distance

    def update_correlated_weights(self, correlation_matrix):
        """
        Updates the weight matrix to account for correlated observations by incorporating the correlation structure
        provided in the correlation matrix.

        Args:
            correlation_matrix (list or numpy.ndarray): The correlation matrix representing the correlation structure of the observations.

        Raises:
            ValueError: If the correlation matrix does not have the same shape as the weight matrix.
        """
        if not isinstance(correlation_matrix, np.ndarray):
            correlation_matrix = np.array(correlation_matrix)

        if correlation_matrix.shape != self.weight_matrix.shape:
            raise ValueError("The correlation matrix must have the same shape as the weight matrix.")

        std_deviations = np.sqrt(np.diag(self.weight_matrix))
        std_mat = np.outer(std_deviations, std_deviations)
        self.weight_matrix = np.divide(correlation_matrix, std_mat)

    def iterative_reweighting(self, max_iterations=10, tolerance=1e-6):
        """
        Performs iterative re-weighting for robust least squares adjustment, updating the weights based on the
        residuals of the previous iteration. This process can help improve the adjustment results in the presence
        of outliers or non-normal errors.

        Args:
            max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 10.
            tolerance (float, optional): The tolerance value for the convergence of the parameter estimates. Defaults to 1e-6.

        Returns:
            numpy.ndarray: The updated parameters after the iterative re-weighting process.
        """
        previous_params = None
        current_params = self.compute_adjustment()['parameters']

        for _ in range(max_iterations):
            residuals = self.observations - self.design_matrix @ current_params
            weights = 1 / (residuals**2 + tolerance)

            self.update_data(weight_matrix=np.diag(weights))
            previous_params = current_params
            current_params = self.compute_adjustment()['parameters']

            if np.linalg.norm(current_params - previous_params) < tolerance:
                break

        return current_params



class BaseLeastSquares:
    """
    A class for performing weighted least squares adjustment on a set of observations.
    
    Attributes:
        y (numpy.ndarray): The observations vector.
        X (numpy.ndarray): The design matrix.
        weight (numpy.ndarray): The weight matrix.
    """
    def __init__(self, X, y, weights=None):
        """
        Initializes the LeastSquaresAdjustment class with observations, design matrix, and an optional weight matrix.
        
        Args:
            y (list): The observations vector.
            A (list): The design matrix.
            weights (list, optional): The weight matrix. Defaults to an identity matrix.
        """
        self.X = X
        self.y = y
        self.weights = weights
        
    @staticmethod
    def validate_input(X, y, weights):
        """
        Validates the input matrices for compatibility.
        
        Args:
            y (list): The observations vector.
            X (list): The design matrix.
            weights (list, optional): The weight matrix.
        """
        if len(y) != len(X):
            raise ValueError("The number of rows in the observations and design matrix must be the same.")

        if weights is not None and len(y) != len(weights):
            raise ValueError("The number of rows in the observations and weight matrix must be the same.")
        
    def update_data(self, X=None, y=None, weights=None):
        """
        Updates the observations, design_matrix, and/or weight_matrix.
        
        Args:
            y (list, optional): The new observations vector.
            X (list, optional): The new design matrix.
            weights (list, optional): The new weight matrix.
        """
        if y is not None:
            self.validate_input(X or self.X, y , weights or self.weights)
            self.y = np.array(y)

        if X is not None:
            self.validate_input( X, y or self.y,  weights or self.weights)
            self.X = np.array(X)

        if weights is not None:
            self.validate_input(X or self.X,y or self.y, weights)
            self.weights = np.array(weights)

    def add_intercept(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def compute_normal_equations(self, X, y, weights=None):
        if weights is not None:
            X = X * np.sqrt(weights)[:, np.newaxis]
            y = y * np.sqrt(weights)
        XtX = X.T @ X
        XtY = X.T @ y
        return XtX, XtY

    def fit(self):
        X_with_intercept = self.add_intercept(self.X)
        XtX, XtY = self.compute_normal_equations(X_with_intercept, self.y, self.weights)
        beta = np.linalg.solve(XtX, XtY)
        intercept = beta[0]
        coefficients = beta[1:]
        return coefficients, intercept
    
    def confidence_interval(self, alpha=0.05):
        # Compute the confidence interval here
        pass

    def r_square(self):
        # Compute the R^2 value here
        pass

    def residuals(self):
        # Compute the residuals here
        pass

    def variance(self):
        # Compute the variance here
        pass

    def covariance_matrix(self):
        # Compute the covariance matrix here
        pass

    def compute_adjustment(self):
        # Compute the adjustment here
        pass

    def iterative_reweighing(self):
        # Perform iterative reweighing here
        pass

    def influence_analyses(self):
        # Perform influence analyses here
        pass

    def error_ellipse(self):
        # Compute the error ellipse here
        pass
    
    def plot_error_ellipse(self, center, a, b, angle, original_points=None):
        # Plot the error ellipse
        pass

class OrdinaryLeastSquares(BaseLeastSquares):
    def __init__(self, X, y):
        super().__init__(X, y, weights=None)

class WeightedLeastSquares(BaseLeastSquares):
    def __init__(self, X, y, weights):
        super().__init__(X, y, weights=weights)

class GeneralizedLeastSquares(BaseLeastSquares):
    def __init__(self, X, y, sigma):
        super().__init__(X, y)
        self.sigma = sigma

    def fit(self):
        if self.sigma is None:
            raise ValueError("Sigma (covariance matrix) must be provided for Generalized Least Squares")
        sigma_inv = np.linalg.inv(self.sigma)
        weights = np.diag(sigma_inv)
        self.weights = weights
        return super().fit()

class TotalLeastSquares(BaseLeastSquares):
    def fit(self):
        X_with_intercept = self.add_intercept(self.X)
        Z = np.hstack((X_with_intercept, self.y.reshape(-1, 1)))
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        V_xy = Vt[-1, :-1]
        V_yy = Vt[-1, -1]
        beta = -V_xy / V_yy
        intercept = beta[0]
        coefficients = beta[1:]
        return coefficients, intercept

class RidgeRegression(BaseLeastSquares):
    def __init__(self, X, y, alpha=1.0):
        super().__init__(X, y)
        self.alpha = alpha

    def fit(self):
        X_with_intercept = self.add_intercept(self.X)
        XtX, XtY = self.compute_normal_equations(X_with_intercept, self.y)
        XtX += self.alpha * np.eye(X_with_intercept.shape[1])
        beta = np.linalg.solve(XtX, XtY)
        intercept = beta[0]
        coefficients = beta[1:]
        return coefficients, intercept
    
class LassoRegression(BaseLeastSquares):
    def __init__(self, X, y, alpha=1.0):
        super().__init__(X, y)
        self.alpha = alpha

    def fit(self):
        model = Lasso(alpha=self.alpha)
        model.fit(self.X, self.y)
        return model.coef_, model.intercept_

class ElasticNetRegression(BaseLeastSquares):
    def __init__(self, X, y, alpha=1.0, l1_ratio=0.5):
        super().__init__(X, y)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self):
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        model.fit(self.X, self.y)
        return model.coef_, model.intercept_

class RobustLeastSquares(BaseLeastSquares):
    def __init__(self, X, y, max_iterations=100, tolerance=1e-6, c=1.345):
        super().__init__(X, y)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.c = c

    def huber_weight(self, x):
        return np.where(np.abs(x) <= self.c, 1, self.c / np.abs(x))

    def fit(self):
        # Implement Robust Least Squares using IRLS with Huber weights
        X = self.X
        y = self.y

        # Add a column of ones for the intercept term
        X_with_intercept = self.add_intercept(X)

        # Initialize the parameters and residuals
        params = np.zeros(X_with_intercept.shape[1])
        residuals = y - X_with_intercept @ params

        for _ in range(self.max_iterations):
            # Compute the Huber weights
            weights = self.huber_weight(residuals)

            # Perform weighted least squares
            self.weights = weights
            coefficients, intercept = super().fit()

            # Update parameters
            new_params = np.hstack((intercept, coefficients))

            # Check for convergence
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            # Update parameters and residuals
            params = new_params
            residuals = y - X_with_intercept @ params

        # Separate the intercept and the other coefficients
        intercept = params[0]
        coefficients = params[1:]
        
        return coefficients, intercept
