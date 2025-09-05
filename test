import sys
import os
import json
from zipfile import BadZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QMessageBox,
    QColorDialog, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QSpinBox, QScrollArea, QFrame, QListWidget, QAbstractItemView
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence

# 檢查 scipy 是否存在，並處理 ImportError
try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: 找不到 scipy 函式庫，平滑曲線功能將不可用。請使用 'pip install scipy' 進行安裝。")

# ==============================================================================
# --- Model: 數據管理核心 ---
# ==============================================================================
class PlottingModel(QObject):
    """
    負責應用程式的所有數據管理、狀態儲存和業務邏輯。
    不直接與 UI 互動，而是透過信號通知變更。
    """
    data_changed = Signal()
    settings_changed = Signal()
    file_loaded = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = []
        self.original_datasets = []
        self.excel_data = None
        self.data_source = 'manual'
        self.settings = self._get_default_settings()
        self.annotation_positions = {}

    def _get_default_settings(self):
        """返回所有繪圖設定的預設值"""
        return {
            "title": "多功能圖表", "x_label": "", "y_label": "",
            "plot_color_hex": "#1f77b4", "bg_color_hex": "#ffffff",
            "major_grid_color_hex": "#cccccc", "minor_grid_color_hex": "#eeeeee",
            "border_color_hex": "#000000", "x_label_color_hex": "#000000",
            "y_label_color_hex": "#000000", "data_label_color_hex": "#000000",
            "minor_tick_color_hex": "#000000", "x_interval": 1.0, "y_interval": 1.0,
            "x_decimal": 2, "y_decimal": 2, "show_data_labels": False,
            "show_x_labels": True, "show_y_labels": True, "data_label_size": 10,
            "is_line_checked": False, "is_scatter_checked": False, "is_bar_checked": False,
            "is_box_checked": False, "line_width": 2.0, "bar_width": 0.8,
            "border_width": 1.0, "point_size": 10.0, "x_label_size": 12,
            "y_label_size": 12, "x_label_bold": False, "y_label_bold": False,
            "x_tick_label_size": 10, "y_tick_label_size": 10,
            "x_tick_label_bold": False, "y_tick_label_bold": False,
            "linestyle": "實線", "marker": "圓形", "connect_scatter": False,
            "show_major_grid": True, "show_minor_grid": False, "axis_border_width": 1.0,
            "legend_size": 10, "tick_direction": "朝外", "minor_x_interval": 0.5,
            "minor_y_interval": 0.5, "minor_tick_length": 4.0, "minor_tick_width": 0.6,
            "major_tick_length": 3.5, "major_tick_width": 0.8, "smooth_line": False,
        }

    def load_file(self, filename):
        """從檔案路徑載入數據"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext == ".xlsx":
                self.excel_data = pd.read_excel(filename, engine="openpyxl")
            elif file_ext == ".xls":
                self.excel_data = pd.read_excel(filename, engine="xlrd")
            elif file_ext == ".csv":
                try:
                    self.excel_data = pd.read_csv(filename, encoding='utf-8')
                except UnicodeDecodeError:
                    self.excel_data = pd.read_csv(filename, encoding='big5')
            else:
                self.error_occurred.emit("不支援的檔案類型，請選擇 .xlsx、.xls 或 .csv 檔案。")
                return

            self.data_source = 'file'
            self.file_loaded.emit(self.excel_data.columns.tolist())
        except BadZipFile:
            self.error_occurred.emit("檔案已損壞或非標準格式。")
        except Exception as e:
            self.error_occurred.emit(f"無法讀取檔案：{e}")

    def update_data_from_file(self, x_col, y_cols):
        """根據選擇的欄位更新數據集"""
        if self.excel_data is None or not x_col or not y_cols:
            self.datasets = []
            self.data_changed.emit()
            return

        try:
            x_data = self.excel_data[x_col].tolist()
            self.datasets = []
            for y_col in y_cols:
                y_data = self.excel_data[y_col].tolist()
                num_points = len(x_data)
                self.datasets.append({
                    'name': y_col, 'x': x_data, 'y': y_data,
                    'colors': [self.settings['plot_color_hex']] * num_points,
                    'primary_color': self.settings['plot_color_hex'],
                    'line_segment_colors': [self.settings['plot_color_hex']] * (num_points - 1)
                })
            self.original_datasets = [ds.copy() for ds in self.datasets]
            self.data_changed.emit()
        except Exception as e:
            self.error_occurred.emit(f"選擇的欄位有問題，無法讀取數據。\n\n詳細錯誤：{e}")
            self.datasets = []
            self.data_changed.emit()

    def update_data_from_table(self, table_data):
        """從表格的數據結構更新內部數據集"""
        self.data_source = 'manual'
        x_data = table_data['x']
        
        new_datasets = []
        for i, y_col_data in enumerate(table_data['y_cols']):
            dataset = {
                'name': y_col_data['name'],
                'x': x_data,
                'y': y_col_data['y'],
                'colors': y_col_data['colors'],
                'primary_color': y_col_data['colors'][0] if y_col_data['colors'] else self.settings['plot_color_hex'],
                'line_segment_colors': [y_col_data['colors'][0]] * (len(x_data) -1) if y_col_data['colors'] else []
            }
            new_datasets.append(dataset)
        
        self.datasets = new_datasets
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.data_changed.emit()
        
    def add_row(self):
        """新增一筆空數據"""
        if not self.datasets:
            self.datasets.append({
                'name': '數據1', 'x': [0], 'y': [0], 
                'colors': [self.settings['plot_color_hex']],
                'primary_color': self.settings['plot_color_hex'],
                'line_segment_colors': []
            })
        else:
            for ds in self.datasets:
                last_x = ds['x'][-1] if ds['x'] else 0
                new_x = last_x + 1 if isinstance(last_x, (int, float)) else 0
                ds['x'].append(new_x)
                ds['y'].append(0)
                ds['colors'].append(ds['primary_color'])
                if len(ds['x']) > 1:
                    ds['line_segment_colors'].append(ds['primary_color'])
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.data_changed.emit()

    def remove_rows(self, row_indices):
        """移除指定索引的數據"""
        for row in sorted(row_indices, reverse=True):
            if self.datasets and 0 <= row < len(self.datasets[0]['x']):
                for dataset in self.datasets:
                    del dataset['x'][row]
                    del dataset['y'][row]
                    del dataset['colors'][row]
                    if row < len(dataset['line_segment_colors']):
                        del dataset['line_segment_colors'][row]
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.data_changed.emit()

    def move_row(self, from_index, to_index):
        """移動數據行的位置"""
        if not (self.datasets and 0 <= from_index < len(self.datasets[0]['x']) and 0 <= to_index < len(self.datasets[0]['x'])):
            return
        for ds in self.datasets:
            ds['x'].insert(to_index, ds['x'].pop(from_index))
            ds['y'].insert(to_index, ds['y'].pop(from_index))
            ds['colors'].insert(to_index, ds['colors'].pop(from_index))
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.data_changed.emit()

    def clear_all(self):
        """清空所有數據和設定"""
        self.datasets = []
        self.original_datasets = []
        self.excel_data = None
        self.data_source = 'manual'
        self.annotation_positions.clear()
        self.settings = self._get_default_settings()
        self.data_changed.emit()
        self.settings_changed.emit()

    def update_setting(self, key, value):
        """更新單一設定值"""
        if key in self.settings and self.settings[key] != value:
            self.settings[key] = value
            self.settings_changed.emit()

    def update_settings(self, new_settings):
        """用新的設定字典更新所有設定"""
        self.settings.update(new_settings)
        self.settings_changed.emit()
    
    def get_settings(self):
        """獲取所有設定"""
        return self.settings.copy()

    def update_point_color(self, ds_index, pt_index, color, artist_type):
        """更新特定數據點的顏色"""
        if ds_index < len(self.datasets):
            if artist_type == 'line':
                if pt_index > 0 and pt_index <= len(self.datasets[ds_index]['line_segment_colors']):
                    self.datasets[ds_index]['line_segment_colors'][pt_index - 1] = color
            elif artist_type in ['scatter', 'bar'] and pt_index < len(self.datasets[ds_index]['colors']):
                self.datasets[ds_index]['colors'][pt_index] = color
            self.data_changed.emit()

    def update_all_colors(self, new_color):
        """更新所有數據點的顏色"""
        self.settings['plot_color_hex'] = new_color
        for ds in self.datasets:
            ds['primary_color'] = new_color
            ds['colors'] = [new_color] * len(ds['colors'])
            ds['line_segment_colors'] = [new_color] * len(ds['line_segment_colors'])
        self.data_changed.emit()
        self.settings_changed.emit()
