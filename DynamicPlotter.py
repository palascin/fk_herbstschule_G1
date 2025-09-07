import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class DynamicPlotter(QWidget):
    def __init__(self, max_points=200, v_max=1.0):
        super().__init__()
        self.setWindowTitle("Skalierbare Live-Plots")
        self.resize(800, 1200)

        layout = QVBoxLayout()
        self.setLayout(layout)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Plot 1: Steering
        self.plot1 = pg.PlotWidget(title="Steering")
        self.curve1 = self.plot1.plot(pen='y')
        splitter.addWidget(self.plot1)

        # Plot 2: Acceleration
        self.plot2 = pg.PlotWidget(title="Acceleration")
        self.curve2 = self.plot2.plot(pen='c')
        splitter.addWidget(self.plot2)

        # Plot 3: Critic Density
        self.plot3 = pg.PlotWidget(title="Critic Density")
        self.curve3 = self.plot3.plot(pen='g')
        self.scatter3 = self.plot3.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=8)
        splitter.addWidget(self.plot3)

        # Plot 4: Velocity with v_max line
        self.plot4 = pg.PlotWidget(title="Velocity")
        self.hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=2))
        self.plot4.addItem(self.hline)
        self.curve4 = self.plot4.plot(pen=pg.mkPen('m', width=2))
        splitter.addWidget(self.plot4)

        splitter.setSizes([300, 300, 300, 300])

        self.max_points = max_points
        self.v_max = v_max
        self.hline.setPos(self.v_max)

        # Data buffers for plots 1-3 (static x-axis)
        self.x = np.linspace(-3, 3, max_points)
        self.x_critic = np.linspace(-6, 2.5, max_points)
        self.y1 = np.zeros(max_points)
        self.y2 = np.zeros(max_points)
        self.y3 = np.zeros(max_points)

        # For scatter points on plot 3
        self.scatter_points = []  # List of x-coordinates for scatter points at y=0

        # For velocity plot (scrolling, only last 100 points)
        self.velocity_buffer = []  # Store all points as (x, y) tuples
        self.current_time = 0.0
        self.dt = 0.1

    def update_plots(self):
        # Update plots 1-3 (unchanged)
        self.curve1.setData(self.x, self.y1)
        self.curve2.setData(self.x, self.y2)
        self.curve3.setData(self.x_critic, self.y3)

        # Update scatter points on plot 3
        if self.scatter_points:
            scatter_y = [0] * len(self.scatter_points)  # All points at y=0
            self.scatter3.setData(self.scatter_points, scatter_y)
        else:
            self.scatter3.setData([], [])

        # Update plot 4 - only show last 100 points
        if len(self.velocity_buffer) > 100:
            recent_points = self.velocity_buffer[-100:]
        else:
            recent_points = self.velocity_buffer

        if recent_points:
            x_vals = [point[0] for point in recent_points]
            y_vals = [point[1] for point in recent_points]
            self.curve4.setData(x_vals, y_vals)

            # Grow from 0 to 10, then shift
            latest_x = x_vals[-1]
            if latest_x <= 10:
                # Still growing: show from 0 to current position (min 10 units)
                self.plot4.setXRange(0, max(10, latest_x), padding=0)
            else:
                # Start shifting: show 10-unit window ending at latest point
                self.plot4.setXRange(latest_x - 10, latest_x, padding=0)
        else:
            # No data yet
            self.curve4.setData([], [])
            self.plot4.setXRange(0, 10, padding=0)

        # Enable auto-range for Y-axis on plots 1-3
        self.plot1.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.plot2.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.plot3.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        # Set Y-range for plot 4 with minimum at 0
        if recent_points:
            y_vals = [point[1] for point in recent_points]
            y_max = max(y_vals)
            y_max = max(y_max, self.v_max * 1.1)  # Ensure v_max line is visible
            self.plot4.setYRange(0, y_max, padding=0.05)
        else:
            # No data yet, show 0 to v_max
            self.plot4.setYRange(0, max(1.0, self.v_max * 1.1), padding=0.05)

        # Update horizontal line position in case v_max changed
        self.hline.setPos(self.v_max)

    def add_point_plot4(self, value: float):
        """Add a new point to the velocity plot"""
        self.velocity_buffer.append((self.current_time, value))
        self.current_time += self.dt

        # Optional: limit buffer size to prevent memory growth
        # Keep more than 100 so we don't lose data when trimming
        if len(self.velocity_buffer) > 500:
            self.velocity_buffer = self.velocity_buffer[-200:]

    def reset_plot4(self):
        """Reset the velocity plot"""
        self.velocity_buffer = []
        self.current_time = 0.0
        self.curve4.setData([], [])
        self.plot4.setXRange(0, 10, padding=0)

    def set_v_max(self, new_v_max: float):
        """Update the v_max value and horizontal line position"""
        self.v_max = new_v_max
        self.hline.setPos(self.v_max)

    def set_scatter_points_plot3(self, x_coords: list):
        """Set scatter points at (x_i, 0) for plot 3"""
        self.scatter_points = list(x_coords)

    def clear_scatter_points_plot3(self):
        """Clear all scatter points from plot 3"""
        self.scatter_points = []

    def set_array_plot1(self, arr: np.ndarray):
        if len(arr) <= self.max_points:
            self.y1 = np.zeros(self.max_points)
            self.y1[:len(arr)] = arr
        else:
            self.y1 = arr[-self.max_points:]

    def set_array_plot2(self, arr: np.ndarray):
        if len(arr) <= self.max_points:
            self.y2 = np.zeros(self.max_points)
            self.y2[:len(arr)] = arr
        else:
            self.y2 = arr[-self.max_points:]

    def set_array_plot3(self, arr: np.ndarray):
        if len(arr) <= self.max_points:
            self.y3 = np.zeros(self.max_points)
            self.y3[:len(arr)] = arr
        else:
            self.y3 = arr[-self.max_points:]