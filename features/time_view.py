from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeViewFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.mqtt_tag = None
        self.initial_buffer_size = 4096  # Initial buffer size
        self.time_view_buffer = deque(maxlen=self.initial_buffer_size)
        self.time_view_timestamps = deque(maxlen=self.initial_buffer_size)
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self.update_time_view_plot)
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.dragging = False
        self.press_x = None
        self.last_data_time = None  # Track time of last data for rate calculation
        self.data_rate = 1.0  # Default samples per second
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"TIME VIEW FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        self.time_widget = QWidget()
        self.time_layout = QVBoxLayout()
        self.time_widget.setLayout(self.time_layout)
        self.time_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        self.time_widget.setMinimumHeight(600)

        tag_layout = QHBoxLayout()
        tag_label = QLabel("Select Tag:")
        tag_label.setStyleSheet("color: white; font-size: 14px;")
        self.tag_combo = QComboBox()
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        if not tags_data:
            self.tag_combo.addItem("No Tags Available")
        else:
            for tag in tags_data:
                self.tag_combo.addItem(tag["tag_name"])
        self.tag_combo.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 5px;")
        self.tag_combo.currentTextChanged.connect(self.setup_time_view_plot)

        # reset_btn = QPushButton("Reset")
        # reset_btn.setStyleSheet("""
        #     QPushButton { background-color: #f39c12; color: white; border: none; padding: 5px; border-radius: 5px; }
        #     QPushButton:hover { background-color: #e67e22; }
        # """)
        # reset_btn.clicked.connect(self.reset_time_view)

        tag_layout.addWidget(tag_label)
        tag_layout.addWidget(self.tag_combo)
        # tag_layout.addWidget(reset_btn)
        tag_layout.addStretch()
        self.time_layout.addLayout(tag_layout)

        self.time_layout.addWidget(self.canvas)

        self.time_result = QTextEdit()
        self.time_result.setReadOnly(True)
        self.time_result.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; padding: 10px;")
        self.time_result.setMinimumHeight(100)
        self.time_result.setText(
            f"Time View for {self.project_name}: Select a tag to start real-time plotting.\n"
            "Buffer adjusts dynamically to data rate."
        )
        self.time_layout.addWidget(self.time_result)
        self.time_layout.addStretch()

        layout.addWidget(self.time_widget)

        if tags_data:
            self.tag_combo.setCurrentIndex(0)
            self.setup_time_view_plot(self.tag_combo.currentText())

    def setup_time_view_plot(self, tag_name):
        if not self.project_name or not tag_name or tag_name == "No Tags Available":
            logging.warning("No project or valid tag selected for Time View!")
            return

        self.mqtt_tag = tag_name
        self.timer.stop()
        self.timer.setInterval(100)  # 100ms updates
        self.time_view_buffer.clear()
        self.time_view_timestamps.clear()
        self.last_data_time = None
        self.data_rate = 1.0  # Reset data rate

        data = self.db.get_tag_values(self.project_name, self.mqtt_tag)
        if data:
            for entry in data[-2:]:
                self.time_view_buffer.extend(entry["values"])
                self.time_view_timestamps.extend([entry["timestamp"]] * len(entry["values"]))

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, color='darkblue')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylabel("Values", rotation=90, labelpad=10)
        self.ax.yaxis.set_label_position("right")
        self.ax.yaxis.tick_right()
        self.ax.set_xlabel("Time (HH:MM:SSS)")
        self.ax.set_xlim(0, 1)  # Initial 1-second window
        self.ax.set_xticks(np.linspace(0, 1, 10))

        self.figure.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.15)
        self.canvas.setMinimumSize(1000, 600)
        self.canvas.draw()
        self.timer.start()

    def adjust_buffer_size(self):
        """Dynamically adjust buffer size based on data rate and window size."""
        xlim = self.ax.get_xlim()
        window_size = xlim[1] - xlim[0]
        if self.data_rate > 0:
            # Buffer should hold at least 2 windows worth of data
            new_buffer_size = max(int(self.data_rate * window_size * 2), 100)  # Minimum 100 points
            if new_buffer_size != self.time_view_buffer.maxlen:
                self.time_view_buffer = deque(self.time_view_buffer, maxlen=new_buffer_size)
                self.time_view_timestamps = deque(self.time_view_timestamps, maxlen=new_buffer_size)
                logging.debug(f"Adjusted buffer size to {new_buffer_size} based on data rate {self.data_rate:.2f} samples/s")

    def generate_y_ticks(self, values):
        if not values or not all(np.isfinite(v) for v in values):
            return np.arange(16390, 46538, 5000)  # Default ticks
        y_max = max(values)
        y_min = min(values)
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
        y_max += padding
        y_min -= padding
        range_val = y_max - y_min
        step = max(range_val / 10, 1)
        step = np.ceil(step / 500) * 500
        ticks = []
        current = np.floor(y_min / step) * step
        while current <= y_max:
            ticks.append(current)
            current += step
        return ticks

    def update_time_view_plot(self):
        if not self.project_name or not self.mqtt_tag:
            self.time_result.setText("No project or tag selected for Time View.")
            return

        current_buffer_size = len(self.time_view_buffer)
        if current_buffer_size < 2:
            self.time_result.setText(
                f"Waiting for sufficient data for {self.mqtt_tag} (Current buffer: {current_buffer_size}/{self.time_view_buffer.maxlen})."
            )
            return

        xlim = self.ax.get_xlim()
        window_size = xlim[1] - xlim[0]

        # Adjust buffer size dynamically
        self.adjust_buffer_size()

        # Calculate samples to fit the window
        samples_per_window = min(current_buffer_size, int(self.data_rate * window_size))
        if samples_per_window < 2:
            samples_per_window = 2

        window_values = list(self.time_view_buffer)[-samples_per_window:]
        window_timestamps = list(self.time_view_timestamps)[-samples_per_window:]

        if not window_values or not all(np.isfinite(v) for v in window_values):
            self.time_result.setText(f"Invalid data received for {self.mqtt_tag}. Buffer: {current_buffer_size}")
            self.ax.set_ylim(16390, 46537)
            self.ax.set_yticks(self.generate_y_ticks([]))
            self.line.set_data([], [])
            self.canvas.draw_idle()
            return

        time_points = np.linspace(xlim[0], xlim[1], samples_per_window)
        self.line.set_data(time_points, window_values)

        y_max = max(window_values)
        y_min = min(window_values)
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
        self.ax.set_ylim(y_min - padding, y_max + padding)
        self.ax.set_yticks(self.generate_y_ticks(window_values))

        if window_timestamps:
            latest_dt = datetime.strptime(window_timestamps[-1], "%Y-%m-%dT%H:%M:%S.%f")
            time_labels = []
            tick_positions = np.linspace(xlim[0], xlim[1], 10)
            for tick in tick_positions:
                delta_seconds = tick * window_size - window_size
                tick_dt = latest_dt + timedelta(seconds=delta_seconds)
                milliseconds = tick_dt.microsecond // 1000
                time_labels.append(f"{tick_dt.strftime('%H:%M:%S:')}{milliseconds:03d}")
            self.ax.set_xticks(tick_positions)
            self.ax.set_xticklabels(time_labels, rotation=0)

        for txt in self.ax.texts:
            txt.remove()

        self.canvas.draw_idle()
        self.time_result.setText(
            f"Time View Data for {self.mqtt_tag}, Latest value: {window_values[-1]:.2f}, "
            f"Window: {window_size:.2f}s, Buffer: {current_buffer_size}/{self.time_view_buffer.maxlen}, "
            f"Data rate: {self.data_rate:.2f} samples/s"
        )

    # def reset_time_view(self):
    #     if hasattr(self, 'ax'):
    #         self.ax.set_xlim(0, 1)
    #         self.ax.set_xticks(np.linspace(0, 1, 10))
    #         self.data_rate = 1.0  # Reset data rate
    #         self.adjust_buffer_size()
    #         self.canvas.draw()
    #         logging.debug("Time View reset to default 1-second window with 10 ticks")

    def on_data_received(self, tag_name, values):
        if tag_name == self.mqtt_tag:
            current_time = datetime.now()
            if self.last_data_time:
                time_delta = (current_time - self.last_data_time).total_seconds()
                if time_delta > 0:
                    self.data_rate = len(values) / time_delta  # Update data rate
            self.last_data_time = current_time

            self.time_view_buffer.extend(values)
            self.time_view_timestamps.extend([current_time.isoformat()] * len(values))
            logging.debug(f"Time View - Received {len(values)} values for {tag_name}, Data rate: {self.data_rate:.2f} samples/s")

    def get_widget(self):
        return self.widget



# from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QTextEdit
# from PyQt5.QtCore import Qt, QTimer
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime, timedelta
# from collections import deque
# import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# class TimeViewFeature:
#     def __init__(self, parent, db, project_name):
#         self.parent = parent
#         self.db = db
#         self.project_name = project_name
#         self.widget = QWidget()
#         self.mqtt_tag = None
#         self.initial_buffer_size = 4096  # Initial buffer size
#         self.time_view_buffer = deque(maxlen=self.initial_buffer_size)
#         self.time_view_timestamps = deque(maxlen=self.initial_buffer_size)
#         self.timer = QTimer(self.widget)
#         self.timer.timeout.connect(self.update_time_view_plot)
#         self.figure = plt.Figure(figsize=(10, 6))
#         self.canvas = FigureCanvas(self.figure)
#         self.last_data_time = None
#         self.data_rate = 1.0  # Default samples per second
#         self.last_window_size = 1.0  # Track last window size for buffer adjustment
#         self.last_data_rate = 1.0  # Track last data rate for buffer adjustment
#         self.text_update_counter = 0  # Counter for less frequent text updates
#         self.initUI()

#     def initUI(self):
#         layout = QVBoxLayout()
#         self.widget.setLayout(layout)

#         header = QLabel(f"TIME VIEW FOR {self.project_name.upper()}")
#         header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
#         layout.addWidget(header, alignment=Qt.AlignCenter)

#         self.time_widget = QWidget()
#         self.time_layout = QVBoxLayout()
#         self.time_widget.setLayout(self.time_layout)
#         self.time_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
#         self.time_widget.setMinimumHeight(600)

#         tag_layout = QHBoxLayout()
#         tag_label = QLabel("Select Tag:")
#         tag_label.setStyleSheet("color: white; font-size: 14px;")
#         self.tag_combo = QComboBox()
#         tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
#         if not tags_data:
#             self.tag_combo.addItem("No Tags Available")
#         else:
#             for tag in tags_data:
#                 self.tag_combo.addItem(tag["tag_name"])
#         self.tag_combo.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 5px;")
#         self.tag_combo.currentTextChanged.connect(self.setup_time_view_plot)

#         tag_layout.addWidget(tag_label)
#         tag_layout.addWidget(self.tag_combo)
#         tag_layout.addStretch()
#         self.time_layout.addLayout(tag_layout)

#         self.time_layout.addWidget(self.canvas)

#         self.time_result = QTextEdit()
#         self.time_result.setReadOnly(True)
#         self.time_result.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; padding: 10px;")
#         self.time_result.setMinimumHeight(100)
#         self.time_result.setText(
#             f"Time View for {self.project_name}: Select a tag to start real-time plotting.\n"
#             "Buffer adjusts dynamically to data rate."
#         )
#         self.time_layout.addWidget(self.time_result)
#         self.time_layout.addStretch()

#         layout.addWidget(self.time_widget)

#         if tags_data:
#             self.tag_combo.setCurrentIndex(0)
#             self.setup_time_view_plot(self.tag_combo.currentText())

#     def setup_time_view_plot(self, tag_name):
#         if not self.project_name or not tag_name or tag_name == "No Tags Available":
#             logging.warning("No project or valid tag selected for Time View!")
#             return

#         self.mqtt_tag = tag_name
#         self.timer.stop()
#         self.time_view_buffer.clear()
#         self.time_view_timestamps.clear()
#         self.last_data_time = None
#         self.data_rate = 1.0
#         self.last_data_rate = 1.0
#         self.last_window_size = 1.0

#         data = self.db.get_tag_values(self.project_name, self.mqtt_tag)
#         if data:
#             for entry in data[-2:]:
#                 self.time_view_buffer.extend(entry["values"])
#                 self.time_view_timestamps.extend([entry["timestamp"]] * len(entry["values"]))

#         self.figure.clear()
#         self.ax = self.figure.add_subplot(111)
#         self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, color='darkblue')
#         self.ax.grid(True, linestyle='--', alpha=0.7)
#         self.ax.set_ylabel("Values", rotation=90, labelpad=10)
#         self.ax.yaxis.set_label_position("right")
#         self.ax.yaxis.tick_right()
#         self.ax.set_xlabel("Time (HH:MM:SSS)")
#         self.ax.set_xlim(0, 1)
#         self.ax.set_xticks(np.linspace(0, 1, 10))
#         self.figure.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.15)
#         self.canvas.setMinimumSize(1000, 600)
#         self.canvas.draw()
#         self.timer.setInterval(100)  # Start with 100ms, adjust dynamically later
#         self.timer.start()

#     def adjust_buffer_size(self, window_size):
#         """Adjust buffer size only when window size or data rate changes significantly."""
#         if abs(window_size - self.last_window_size) > 0.1 or abs(self.data_rate - self.last_data_rate) > 0.5:
#             new_buffer_size = max(int(self.data_rate * window_size * 2), 100)
#             if new_buffer_size != self.time_view_buffer.maxlen:
#                 self.time_view_buffer = deque(self.time_view_buffer, maxlen=new_buffer_size)
#                 self.time_view_timestamps = deque(self.time_view_timestamps, maxlen=new_buffer_size)
#                 logging.debug(f"Adjusted buffer size to {new_buffer_size} based on data rate {self.data_rate:.2f} samples/s")
#             self.last_window_size = window_size
#             self.last_data_rate = self.data_rate

#     def generate_y_ticks(self, values):
#         if not values.size:
#             return np.arange(16390, 46538, 5000)
#         y_max, y_min = np.max(values), np.min(values)
#         padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
#         y_max += padding
#         y_min -= padding
#         range_val = y_max - y_min
#         step = max(range_val / 10, 1)
#         step = np.ceil(step / 500) * 500
#         return np.arange(np.floor(y_min / step) * step, y_max + step, step)

#     def update_time_view_plot(self):
#         if not self.project_name or not self.mqtt_tag or len(self.time_view_buffer) < 2:
#             return

#         xlim = self.ax.get_xlim()
#         window_size = xlim[1] - xlim[0]
#         self.adjust_buffer_size(window_size)

#         samples_per_window = min(len(self.time_view_buffer), max(int(self.data_rate * window_size), 2))
#         window_values = np.array(list(self.time_view_buffer)[-samples_per_window:])
#         window_timestamps = list(self.time_view_timestamps)[-samples_per_window:]

#         if not window_values.size:
#             return

#         time_points = np.linspace(xlim[0], xlim[1], samples_per_window)
#         self.line.set_data(time_points, window_values)

#         y_min, y_max = np.min(window_values), np.max(window_values)
#         padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
#         self.ax.set_ylim(y_min - padding, y_max + padding)
#         self.ax.set_yticks(self.generate_y_ticks(window_values))

#         # Update time labels and text less frequently (every ~1s)
#         if self.text_update_counter % 10 == 0:
#             latest_dt = datetime.strptime(window_timestamps[-1], "%Y-%m-%dT%H:%M:%S.%f")
#             tick_positions = np.linspace(xlim[0], xlim[1], 10)
#             time_labels = [
#                 f"{(latest_dt + timedelta(seconds=tick * window_size - window_size)).strftime('%H:%M:%S:')}"
#                 f"{(latest_dt + timedelta(seconds=tick * window_size - window_size)).microsecond // 1000:03d}"
#                 for tick in tick_positions
#             ]
#             self.ax.set_xticks(tick_positions)
#             self.ax.set_xticklabels(time_labels, rotation=0)
#             self.time_result.setText(
#                 f"Time View Data for {self.mqtt_tag}, Latest value: {window_values[-1]:.2f}, "
#                 f"Window: {window_size:.2f}s, Buffer: {len(self.time_view_buffer)}/{self.time_view_buffer.maxlen}, "
#                 f"Data rate: {self.data_rate:.2f} samples/s"
#             )

#         self.canvas.draw()  # Faster than draw_idle()
#         self.text_update_counter += 1

#     def on_data_received(self, tag_name, values):
#         if tag_name == self.mqtt_tag:
#             current_time = datetime.now()
#             if self.last_data_time:
#                 time_delta = (current_time - self.last_data_time).total_seconds()
#                 if time_delta > 0:
#                     self.data_rate = len(values) / time_delta
#                     # Dynamically adjust timer interval based on data rate
#                     self.timer.setInterval(max(50, int(1000 / self.data_rate / 2)))  # Min 50ms
#             self.last_data_time = current_time
#             self.time_view_buffer.extend(values)
#             self.time_view_timestamps.extend([current_time.isoformat()] * len(values))
#             logging.debug(f"Time View - Received {len(values)} values for {tag_name}, Data rate: {self.data_rate:.2f} samples/s")

#     def get_widget(self):
#         return self.widget