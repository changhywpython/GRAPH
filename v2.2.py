import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QMessageBox,
    QColorDialog, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QSpinBox, QScrollArea, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
import os
from zipfile import BadZipFile
import json
import numpy as np

# 檢查 scipy 是否存在，並處理 ImportError
try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: 找不到 scipy 函式庫，平滑曲線功能將不可用。請使用 'pip install scipy' 進行安裝。")

class PlottingApp(QMainWindow):
    """
    主要應用程式視窗類別，包含 GUI 和所有繪圖邏輯。
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("多功能繪圖工具-V2.2")
        self.setGeometry(100, 100, 1200, 800)
        
        # 設置 Matplotlib 字體以支援中文顯示
        self.set_matplotlib_font()
        
        # Matplotlib 圖形設定
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.legend = None

        self.excel_data = None
        self.x_data = []
        self.y_data = []
        self.colors_data = []
        self.data_source = 'manual'
        self.selected_point_index = -1
        self.is_updating_table = False

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)

        # 預設顏色設定
        self.plot_color_hex = "#1f77b4"
        self.bg_color_hex = "#ffffff"
        self.major_grid_color_hex = "#cccccc"
        self.minor_grid_color_hex = "#eeeeee"
        self.border_color_hex = "#000000"
        self.x_label_color_hex = "#000000"
        self.y_label_color_hex = "#000000"
        self.data_label_color_hex = "#000000"
        self.minor_tick_color_hex = "#000000"

        self.init_ui()

        # 連結滑鼠點擊事件
        self.canvas.mpl_connect('button_press_event', self.on_point_click)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()

    def set_matplotlib_font(self):
        """
        嘗試設定中文字體，如果失敗則印出警告。
        """
        try:
            font_names = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti TC', 'Arial Unicode MS']
            found_font = None
            for font_name in font_names:
                if fm.fontManager.findfont(font_name, fallback_to_default=False):
                    found_font = font_name
                    break
            
            if found_font:
                plt.rcParams['font.sans-serif'] = found_font
                print(f"找到並使用中文字體: {found_font}")
            else:
                font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                for font_path in font_paths:
                    if any(kw in os.path.basename(font_path).lower() for kw in ['simhei', 'yahei', 'pingfang', 'heiti']):
                        fm.fontManager.addfont(font_path)
                        plt.rcParams['font.sans-serif'] = os.path.basename(font_path).replace('.ttf', '')
                        print(f"找到並使用系統字體: {plt.rcParams['font.sans-serif'][0]}")
                        break

        except Exception as e:
            print(f"警告: 設定中文字體失敗，可能會出現亂碼。錯誤訊息: {e}")
        
        plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
    
    def init_ui(self):
        """
        初始化 GUI 介面元件。
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左側控制面板
        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(400)
        
        self.data_settings_tab = QWidget()
        self.plot_settings_tab = QWidget()
        
        self.control_tabs.addTab(self.data_settings_tab, "數據與檔案")
        self.control_tabs.addTab(self.plot_settings_tab, "繪圖設定")

        main_layout.addWidget(self.control_tabs)

        # 繪圖區域 (右側)
        plot_area_widget = QWidget()
        plot_area_layout = QVBoxLayout(plot_area_widget)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_area_layout.addWidget(self.toolbar)
        plot_area_layout.addWidget(self.canvas)
        
        main_layout.addWidget(plot_area_widget)

        # 數據與檔案分頁的內容
        data_settings_layout = QVBoxLayout(self.data_settings_tab)
        data_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        excel_layout = QGridLayout()
        self.load_excel_btn = QPushButton("選擇 Excel 或 CSV 檔案")
        self.load_excel_btn.clicked.connect(self.load_excel_file)
        excel_layout.addWidget(self.load_excel_btn, 0, 0, 1, 2)
        
        self.x_col_label = QLabel("X 欄位:")
        excel_layout.addWidget(self.x_col_label, 1, 0)
        self.x_col_combo = QComboBox()
        self.x_col_combo.currentIndexChanged.connect(self.update_data_from_file_input)
        excel_layout.addWidget(self.x_col_combo, 1, 1)
        
        self.y_col_label = QLabel("Y 欄位:")
        excel_layout.addWidget(self.y_col_label, 2, 0)
        self.y_col_combo = QComboBox()
        self.y_col_combo.currentIndexChanged.connect(self.update_data_from_file_input)
        excel_layout.addWidget(self.y_col_combo, 2, 1)
        excel_group = self.create_collapsible_container("檔案讀取", excel_layout)
        data_settings_layout.addWidget(excel_group)
        
        data_table_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["X 數據", "Y 數據", "顏色"])
        self.data_table.setColumnWidth(2, 60)
        self.data_table.itemChanged.connect(self.update_data_from_table)
        self.data_table.cellClicked.connect(self.pick_color_for_cell)
        data_table_layout.addWidget(self.data_table)

        table_control_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("新增行")
        self.add_row_btn.clicked.connect(self.add_row)
        self.remove_row_btn = QPushButton("刪除行")
        self.remove_row_btn.clicked.connect(self.remove_row)
        self.move_up_btn = QPushButton("上移")
        self.move_up_btn.clicked.connect(self.move_row_up)
        self.move_down_btn = QPushButton("下移")
        self.move_down_btn.clicked.connect(self.move_row_down)

        table_control_layout.addWidget(self.add_row_btn)
        table_control_layout.addWidget(self.remove_row_btn)
        table_control_layout.addWidget(self.move_up_btn)
        table_control_layout.addWidget(self.move_down_btn)
        data_table_layout.addLayout(table_control_layout)
        data_table_group = self.create_collapsible_container("數據表格", data_table_layout)
        data_settings_layout.addWidget(data_table_group)

        # 繪圖設定分頁的內容
        plot_settings_scroll_area = QScrollArea()
        plot_settings_scroll_area.setWidgetResizable(True)
        plot_settings_content_widget = QWidget()
        plot_settings_layout = QVBoxLayout(plot_settings_content_widget)
        plot_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 圖表類型選擇 (同一行)
        plot_type_layout = QHBoxLayout()
        self.line_checkbox = QCheckBox("折線圖")
        self.scatter_checkbox = QCheckBox("散佈圖")
        self.bar_checkbox = QCheckBox("長條圖")
        self.line_checkbox.toggled.connect(self.toggle_plot_settings)
        self.scatter_checkbox.toggled.connect(self.toggle_plot_settings)
        self.bar_checkbox.toggled.connect(self.toggle_plot_settings)
        plot_type_layout.addWidget(self.line_checkbox)
        plot_type_layout.addWidget(self.scatter_checkbox)
        plot_type_layout.addWidget(self.bar_checkbox)
        plot_type_group = self.create_collapsible_container("圖表類型", plot_type_layout)
        plot_settings_layout.addWidget(plot_type_group)

        # 繪圖參數設定 (調整為緊湊佈局)
        self.settings_layout = QVBoxLayout()
        self.settings_layout.addWidget(QLabel("圖表標題:"))
        self.title_input = QLineEdit()
        self.title_input.textChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.title_input)

        self.settings_layout.addWidget(QLabel("X 軸標籤:"))
        self.x_label_input = QLineEdit()
        self.x_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
        self.x_label_input.textChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.x_label_input)

        self.settings_layout.addWidget(QLabel("Y 軸標籤:"))
        self.y_label_input = QLineEdit()
        self.y_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
        self.y_label_input.textChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.y_label_input)
        
        self.settings_layout.addWidget(QLabel("圖例名稱:"))
        self.legend_input = QLineEdit()
        self.legend_input.setPlaceholderText("圖例名稱 (選填)")
        self.legend_input.textChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.legend_input)

        self.settings_layout.addWidget(QLabel("圖例文字大小:"))
        self.legend_size_spinbox = QSpinBox()
        self.legend_size_spinbox.setMinimum(1)
        self.legend_size_spinbox.setValue(10)
        self.legend_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.legend_size_spinbox)
        
        # 數據標籤顯示選項
        self.settings_layout.addWidget(QLabel("顯示數據值:"))
        data_label_layout = QHBoxLayout()
        self.show_data_labels_checkbox = QCheckBox("顯示數據值")
        self.show_data_labels_checkbox.setChecked(False)
        self.show_data_labels_checkbox.toggled.connect(self.update_plot_with_timer)
        data_label_layout.addWidget(self.show_data_labels_checkbox)
        
        self.show_x_labels_checkbox = QCheckBox("X")
        self.show_x_labels_checkbox.setChecked(True)
        self.show_x_labels_checkbox.toggled.connect(self.update_plot_with_timer)
        data_label_layout.addWidget(self.show_x_labels_checkbox)
        
        self.show_y_labels_checkbox = QCheckBox("Y")
        self.show_y_labels_checkbox.setChecked(True)
        self.show_y_labels_checkbox.toggled.connect(self.update_plot_with_timer)
        data_label_layout.addWidget(self.show_y_labels_checkbox)
        self.settings_layout.addLayout(data_label_layout)
        
        self.settings_layout.addWidget(QLabel("數據標籤大小:"))
        self.data_label_size_spinbox = QSpinBox()
        self.data_label_size_spinbox.setMinimum(1)
        self.data_label_size_spinbox.setValue(10)
        self.data_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        self.settings_layout.addWidget(self.data_label_size_spinbox)
        
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("X軸間隔:"))
        self.x_interval_spinbox = QDoubleSpinBox()
        self.x_interval_spinbox.setMinimum(0.0)
        self.x_interval_spinbox.setMaximum(float('inf'))
        self.x_interval_spinbox.setValue(1.0)
        self.x_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        interval_layout.addWidget(self.x_interval_spinbox)
        self.settings_layout.addLayout(interval_layout)

        interval_layout_y = QHBoxLayout()
        interval_layout_y.addWidget(QLabel("Y軸間隔:"))
        self.y_interval_spinbox = QDoubleSpinBox()
        self.y_interval_spinbox.setMinimum(0.0)
        self.y_interval_spinbox.setMaximum(float('inf'))
        self.y_interval_spinbox.setValue(1.0)
        self.y_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        interval_layout_y.addWidget(self.y_interval_spinbox)
        self.settings_layout.addLayout(interval_layout_y)

        decimal_layout = QHBoxLayout()
        decimal_layout.addWidget(QLabel("X軸小數點:"))
        self.x_decimal_spinbox = QSpinBox()
        self.x_decimal_spinbox.setMinimum(0)
        self.x_decimal_spinbox.setMaximum(999999999)
        self.x_decimal_spinbox.setValue(2)
        self.x_decimal_spinbox.valueChanged.connect(self.update_plot_with_timer)
        decimal_layout.addWidget(self.x_decimal_spinbox)
        self.settings_layout.addLayout(decimal_layout)
        
        decimal_layout_y = QHBoxLayout()
        decimal_layout_y.addWidget(QLabel("Y軸小數點:"))
        self.y_decimal_spinbox = QSpinBox()
        self.y_decimal_spinbox.setMinimum(0)
        self.y_decimal_spinbox.setMaximum(999999999)
        self.y_decimal_spinbox.setValue(2)
        self.y_decimal_spinbox.valueChanged.connect(self.update_plot_with_timer)
        decimal_layout_y.addWidget(self.y_decimal_spinbox)
        self.settings_layout.addLayout(decimal_layout_y)

        self.settings_group = self.create_collapsible_container("圖表設定", self.settings_layout)
        plot_settings_layout.addWidget(self.settings_group)
        
        # 動態顯示的繪圖樣式設定
        self.style_layout = QVBoxLayout()
        self.line_width_spinbox = QDoubleSpinBox()
        self.bar_width_spinbox = QDoubleSpinBox()
        self.point_size_spinbox = QDoubleSpinBox()
        self.linestyle_combo = QComboBox()
        self.marker_combo = QComboBox()
        self.smooth_line_checkbox = QCheckBox("平滑曲線") # 新增平滑曲線選項
        self.smooth_line_checkbox.toggled.connect(self.update_plot_with_timer)
        self.setup_dynamic_widgets()
        
        plot_color_layout = QHBoxLayout()
        plot_color_layout.addWidget(QLabel("圖表顏色:"))
        self.plot_color_btn = QPushButton("選擇顏色")
        self.plot_color_btn.clicked.connect(lambda: self.pick_color("plot"))
        plot_color_layout.addWidget(self.plot_color_btn)
        self.style_layout.addLayout(plot_color_layout)

        border_width_layout = QHBoxLayout()
        border_width_layout.addWidget(QLabel("邊框粗度:"))
        self.border_width_spinbox = QDoubleSpinBox()
        self.border_width_spinbox.setMinimum(0.0)
        self.border_width_spinbox.setMaximum(float('inf'))
        self.border_width_spinbox.setValue(1.0)
        self.border_width_spinbox.setSingleStep(0.5)
        self.border_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        border_width_layout.addWidget(self.border_width_spinbox)
        self.style_layout.addLayout(border_width_layout)

        border_color_layout = QHBoxLayout()
        border_color_layout.addWidget(QLabel("邊框顏色:"))
        self.border_color_btn = QPushButton("選擇顏色")
        self.border_color_btn.clicked.connect(lambda: self.pick_color("border"))
        border_color_layout.addWidget(self.border_color_btn)
        self.style_layout.addLayout(border_color_layout)

        self.style_group = self.create_collapsible_container("繪圖樣式設定", self.style_layout)
        plot_settings_layout.addWidget(self.style_group)
        
        # 背景與座標軸設定
        axis_style_layout = QVBoxLayout()
        
        axis_style_layout.addWidget(QLabel("背景顏色:"))
        self.bg_color_btn = QPushButton("選擇顏色")
        self.bg_color_btn.clicked.connect(lambda: self.pick_color("background"))
        axis_style_layout.addWidget(self.bg_color_btn)
        
        axis_style_layout.addWidget(QLabel("X 軸標籤:"))
        x_label_props_layout = QHBoxLayout()
        x_label_props_layout.addWidget(QLabel("大小:"))
        self.x_label_size_spinbox = QSpinBox()
        self.x_label_size_spinbox.setMinimum(1)
        self.x_label_size_spinbox.setValue(12)
        self.x_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        x_label_props_layout.addWidget(self.x_label_size_spinbox)
        self.x_label_bold_checkbox = QCheckBox("粗體")
        self.x_label_bold_checkbox.toggled.connect(self.update_plot_with_timer)
        x_label_props_layout.addWidget(self.x_label_bold_checkbox)
        self.x_label_color_btn = QPushButton("顏色")
        self.x_label_color_btn.clicked.connect(lambda: self.pick_color("x_label"))
        x_label_props_layout.addWidget(self.x_label_color_btn)
        axis_style_layout.addLayout(x_label_props_layout)

        axis_style_layout.addWidget(QLabel("Y 軸標籤:"))
        y_label_props_layout = QHBoxLayout()
        y_label_props_layout.addWidget(QLabel("大小:"))
        self.y_label_size_spinbox = QSpinBox()
        self.y_label_size_spinbox.setMinimum(1)
        self.y_label_size_spinbox.setValue(12)
        self.y_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        y_label_props_layout.addWidget(self.y_label_size_spinbox)
        self.y_label_bold_checkbox = QCheckBox("粗體")
        self.y_label_bold_checkbox.toggled.connect(self.update_plot_with_timer)
        y_label_props_layout.addWidget(self.y_label_bold_checkbox)
        self.y_label_color_btn = QPushButton("顏色")
        self.y_label_color_btn.clicked.connect(lambda: self.pick_color("y_label"))
        y_label_props_layout.addWidget(self.y_label_color_btn)
        axis_style_layout.addLayout(y_label_props_layout)
        
        tick_size_layout = QHBoxLayout()
        tick_size_layout.addWidget(QLabel("X軸刻度大小:"))
        self.x_tick_label_size_spinbox = QSpinBox()
        self.x_tick_label_size_spinbox.setMinimum(1)
        self.x_tick_label_size_spinbox.setValue(10)
        self.x_tick_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        tick_size_layout.addWidget(self.x_tick_label_size_spinbox)
        axis_style_layout.addLayout(tick_size_layout)

        tick_size_layout_y = QHBoxLayout()
        tick_size_layout_y.addWidget(QLabel("Y軸刻度大小:"))
        self.y_tick_label_size_spinbox = QSpinBox()
        self.y_tick_label_size_spinbox.setMinimum(1)
        self.y_tick_label_size_spinbox.setValue(10)
        self.y_tick_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        tick_size_layout_y.addWidget(self.y_tick_label_size_spinbox)
        axis_style_layout.addLayout(tick_size_layout_y)

        axis_border_width_layout = QHBoxLayout()
        axis_border_width_layout.addWidget(QLabel("座標軸邊框粗度:"))
        self.axis_border_width_spinbox = QDoubleSpinBox()
        self.axis_border_width_spinbox.setMinimum(0.0)
        self.axis_border_width_spinbox.setMaximum(float('inf'))
        self.axis_border_width_spinbox.setValue(1.0)
        self.axis_border_width_spinbox.setSingleStep(0.5)
        self.axis_border_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        axis_border_width_layout.addWidget(self.axis_border_width_spinbox)
        axis_style_layout.addLayout(axis_border_width_layout)

        self.tick_direction_layout = QHBoxLayout()
        self.tick_direction_layout.addWidget(QLabel("刻度線朝向:"))
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(["朝外", "朝內", "朝內外"])
        self.tick_direction_combo.currentIndexChanged.connect(self.update_plot_with_timer)
        self.tick_direction_layout.addWidget(self.tick_direction_combo)
        axis_style_layout.addLayout(self.tick_direction_layout)

        axis_style_group = self.create_collapsible_container("背景與座標軸設定", axis_style_layout)
        plot_settings_layout.addWidget(axis_style_group)
        
        # 網格線設定
        grid_layout = QVBoxLayout()
        
        self.major_grid_checkbox = QCheckBox("顯示主網格線")
        self.major_grid_checkbox.setChecked(True)
        self.major_grid_checkbox.toggled.connect(self.update_plot_with_timer)
        grid_layout.addWidget(self.major_grid_checkbox)
        
        self.major_grid_color_btn = QPushButton("主網格顏色")
        self.major_grid_color_btn.clicked.connect(lambda: self.pick_color("major_grid"))
        grid_layout.addWidget(self.major_grid_color_btn)

        self.minor_grid_checkbox = QCheckBox("顯示次網格線")
        self.minor_grid_checkbox.setChecked(False)
        self.minor_grid_checkbox.toggled.connect(self.update_plot_with_timer)
        grid_layout.addWidget(self.minor_grid_checkbox)
        
        self.minor_grid_color_btn = QPushButton("次網格顏色")
        self.minor_grid_color_btn.clicked.connect(lambda: self.pick_color("minor_grid"))
        grid_layout.addWidget(self.minor_grid_color_btn)
        
        grid_group = self.create_collapsible_container("網格線設定", grid_layout)
        plot_settings_layout.addWidget(grid_group)

        # 次刻度線設定
        minor_tick_layout = QVBoxLayout()
        minor_tick_layout.addWidget(QLabel("次刻度線間隔:"))
        self.minor_x_interval_spinbox = QDoubleSpinBox()
        self.minor_x_interval_spinbox.setMinimum(0.0)
        self.minor_x_interval_spinbox.setMaximum(float('inf'))
        self.minor_x_interval_spinbox.setValue(0.5)
        self.minor_x_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        minor_tick_layout.addWidget(self.minor_x_interval_spinbox)
        
        self.minor_y_interval_spinbox = QDoubleSpinBox()
        self.minor_y_interval_spinbox.setMinimum(0.0)
        self.minor_y_interval_spinbox.setMaximum(float('inf'))
        self.minor_y_interval_spinbox.setValue(0.5)
        self.minor_y_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        minor_tick_layout.addWidget(self.minor_y_interval_spinbox)

        minor_tick_layout.addWidget(QLabel("次刻度線長度:"))
        self.minor_tick_length_spinbox = QDoubleSpinBox()
        self.minor_tick_length_spinbox.setMinimum(0.0)
        self.minor_tick_length_spinbox.setValue(4.0)
        self.minor_tick_length_spinbox.valueChanged.connect(self.update_plot_with_timer)
        minor_tick_layout.addWidget(self.minor_tick_length_spinbox)

        self.minor_tick_group = self.create_collapsible_container("次刻度線設定", minor_tick_layout)
        plot_settings_layout.addWidget(self.minor_tick_group)

        template_layout = QHBoxLayout()
        self.save_template_btn = QPushButton("儲存設定為範本")
        self.save_template_btn.clicked.connect(self.save_template)
        template_layout.addWidget(self.save_template_btn)
        self.load_template_btn = QPushButton("載入範本")
        self.load_template_btn.clicked.connect(self.load_template)
        template_layout.addWidget(self.load_template_btn)
        template_group = self.create_collapsible_container("範本功能", template_layout)
        plot_settings_layout.addWidget(template_group)
        
        clear_button = QPushButton("清空圖表")
        clear_button.clicked.connect(self.clear_plot)
        clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; font-weight: bold;")
        plot_settings_layout.addWidget(clear_button)
        
        plot_settings_layout.addStretch()

        plot_settings_scroll_area.setWidget(plot_settings_content_widget)
        main_plot_settings_layout = QVBoxLayout(self.plot_settings_tab)
        main_plot_settings_layout.addWidget(plot_settings_scroll_area)

        self.update_button_color()
        self.update_plot()
        self.update_table()
        self.toggle_plot_settings() # 初始化介面顯示

    def setup_dynamic_widgets(self):
        """
        為動態顯示的組件建立並添加布局。
        """
        self.style_layout.addWidget(QLabel("線條寬度:"))
        self.line_width_spinbox.setMinimum(0.5)
        self.line_width_spinbox.setMaximum(float('inf'))
        self.line_width_spinbox.setValue(2.0)
        self.line_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        self.style_layout.addWidget(self.line_width_spinbox)
        
        self.style_layout.addWidget(QLabel("長條圖寬度:"))
        self.bar_width_spinbox.setMinimum(0.01)
        self.bar_width_spinbox.setMaximum(float('inf')) # 長條圖寬度最大值改為無窮
        self.bar_width_spinbox.setSingleStep(0.05)
        self.bar_width_spinbox.setValue(0.8)
        self.bar_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        self.style_layout.addWidget(self.bar_width_spinbox)
        
        self.style_layout.addWidget(QLabel("點的大小:"))
        self.point_size_spinbox.setMinimum(1.0)
        self.point_size_spinbox.setMaximum(float('inf'))
        self.point_size_spinbox.setValue(10.0)
        self.point_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        self.style_layout.addWidget(self.point_size_spinbox)

        self.style_layout.addWidget(QLabel("線條樣式:"))
        self.linestyle_combo.addItems(["實線", "虛線", "點虛線", "點"])
        self.linestyle_combo.currentIndexChanged.connect(self.update_plot_with_timer)
        self.style_layout.addWidget(self.linestyle_combo)
        
        self.style_layout.addWidget(QLabel("標記樣式:"))
        self.marker_combo.addItems(["圓形", "方形", "三角形", "星形", "無"])
        self.marker_combo.currentIndexChanged.connect(self.update_plot_with_timer)
        self.style_layout.addWidget(self.marker_combo)
        
        # 只有在 scipy 存在時才顯示平滑曲線選項
        if SCIPY_AVAILABLE:
            self.style_layout.addWidget(self.smooth_line_checkbox)

    def toggle_plot_settings(self):
        """
        根據選定的圖表類型，動態顯示/隱藏相關的設定。
        """
        is_line_checked = self.line_checkbox.isChecked()
        is_scatter_checked = self.scatter_checkbox.isChecked()
        is_bar_checked = self.bar_checkbox.isChecked()

        # 折線圖設定
        self.line_width_spinbox.setVisible(is_line_checked)
        self.linestyle_combo.setVisible(is_line_checked)
        self.smooth_line_checkbox.setVisible(is_line_checked and SCIPY_AVAILABLE) # 增加 scipy 檢查
        
        # 散佈圖與折線圖共用標記設定
        self.point_size_spinbox.setVisible(is_scatter_checked or is_line_checked)
        self.marker_combo.setVisible(is_scatter_checked or is_line_checked)

        # 長條圖設定
        self.bar_width_spinbox.setVisible(is_bar_checked)
        
        # 使用 QWidget 隱藏相關的 QLabel
        self.line_width_spinbox.parentWidget().layout().itemAt(0).widget().setVisible(is_line_checked)
        self.bar_width_spinbox.parentWidget().layout().itemAt(0).widget().setVisible(is_bar_checked)
        self.point_size_spinbox.parentWidget().layout().itemAt(0).widget().setVisible(is_scatter_checked or is_line_checked)
        self.linestyle_combo.parentWidget().layout().itemAt(0).widget().setVisible(is_line_checked)
        self.marker_combo.parentWidget().layout().itemAt(0).widget().setVisible(is_scatter_checked or is_line_checked)
        
        self.update_plot_with_timer()

    def on_point_click(self, event):
        """
        處理圖表上的滑鼠點擊事件，並高亮對應的表格行。
        """
        if event.button == 1 and event.inaxes:
            
            # 遍歷所有的子圖
            for artist in self.ax.get_children():
                if isinstance(artist, plt.matplotlib.collections.PathCollection) or isinstance(artist, plt.matplotlib.lines.Line2D):
                    contains, _ = artist.contains(event)
                    if contains:
                        # 找到被點選的點的索引
                        ind = _.get("ind")[0]
                        self.selected_point_index = ind
                        
                        # 清除舊的高亮
                        self.data_table.clearSelection()
                        # 高亮新選定的行
                        self.data_table.selectRow(ind)
                        
                        # 在控制台輸出訊息
                        print(f"選中了點: (X: {self.x_data[ind]}, Y: {self.y_data[ind]})")
                        return

            # 如果沒有點被選中，重置選中狀態
            self.selected_point_index = -1
            self.data_table.clearSelection()

    def update_button_color(self):
        """
        更新顏色選擇按鈕的背景色以反映當前顏色。
        """
        self.plot_color_btn.setStyleSheet(f"background-color: {self.plot_color_hex};")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color_hex};")
        self.major_grid_color_btn.setStyleSheet(f"background-color: {self.major_grid_color_hex};")
        self.minor_grid_color_btn.setStyleSheet(f"background-color: {self.minor_grid_color_hex};")
        self.border_color_btn.setStyleSheet(f"background-color: {self.border_color_hex};")
        self.x_label_color_btn.setStyleSheet(f"background-color: {self.x_label_color_hex};")
        self.y_label_color_btn.setStyleSheet(f"background-color: {self.y_label_color_hex};")
        
    def pick_color(self, target):
        """
        開啟調色盤，讓使用者選擇顏色。
        """
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            if target == "plot":
                if self.selected_point_index != -1 and self.selected_point_index < len(self.colors_data):
                    self.colors_data[self.selected_point_index] = hex_color
                    self.update_table()
                else:
                    self.plot_color_hex = hex_color
            elif target == "background":
                self.bg_color_hex = hex_color
            elif target == "major_grid":
                self.major_grid_color_hex = hex_color
            elif target == "minor_grid":
                self.minor_grid_color_hex = hex_color
            elif target == "border":
                self.border_color_hex = hex_color
            elif target == "x_label":
                self.x_label_color_hex = hex_color
            elif target == "y_label":
                self.y_label_color_hex = hex_color
            
            self.update_button_color()
            self.update_plot_with_timer()

    def pick_color_for_cell(self, row, col):
        """
        開啟調色盤，讓使用者為特定單元格選擇顏色。
        """
        if col == 2:
            color = QColorDialog.getColor()
            if color.isValid():
                hex_color = color.name()
                item = QTableWidgetItem(hex_color)
                item.setBackground(QColor(hex_color))
                self.is_updating_table = True
                self.data_table.setItem(row, col, item)
                self.is_updating_table = False
                self.update_data_from_table()
                
    def update_plot_with_timer(self):
        """
        透過計時器觸發繪圖，以避免頻繁更新。
        """
        self.update_timer.start(300)

    def update_data_from_table(self):
        """
        從表格更新數據並設定數據來源。
        """
        if self.is_updating_table:
            return

        self.data_source = 'manual'
        new_x_data = []
        new_y_data = []
        new_colors_data = []
        
        for row in range(self.data_table.rowCount()):
            x_item = self.data_table.item(row, 0)
            y_item = self.data_table.item(row, 1)
            color_item = self.data_table.item(row, 2)

            x_value = x_item.text() if x_item else "0"
            y_value = y_item.text() if y_item else "0"
            color_value = color_item.text() if color_item else self.plot_color_hex
            
            try:
                new_x_data.append(float(x_value))
            except (ValueError, TypeError):
                new_x_data.append(x_value)
            try:
                new_y_data.append(float(y_value))
            except (ValueError, TypeError):
                new_y_data.append(y_value)
                
            new_colors_data.append(color_value)
                
        self.x_data = new_x_data
        self.y_data = new_y_data
        self.colors_data = new_colors_data
        
        self.update_plot_with_timer()

    def update_data_from_file_input(self):
        """
        從檔案欄位選單更新數據並設定數據來源。
        """
        if self.excel_data is None:
            return

        self.data_source = 'file'
        x_col_name = self.x_col_combo.currentText()
        y_col_name = self.y_col_combo.currentText()

        if not x_col_name or not y_col_name:
            self.x_data = []
            self.y_data = []
            self.colors_data = []
            self.update_plot_with_timer()
            self.update_table()
            return

        try:
            self.x_data = self.excel_data[x_col_name].tolist()
            self.y_data = self.excel_data[y_col_name].tolist()
            self.colors_data = [self.plot_color_hex] * len(self.x_data)
            
            if not self.x_label_input.text():
                self.x_label_input.setText(x_col_name)
            if not self.y_label_input.text():
                self.y_label_input.setText(y_col_name)

            self.update_plot_with_timer()
            self.update_table()
        except Exception as e:
            self.x_data = []
            self.y_data = []
            self.colors_data = []
            self.ax.clear()
            QMessageBox.critical(self, "數據讀取錯誤", f"選擇的欄位有問題，無法讀取數據。\n\n詳細錯誤：{e}")
            self.ax.set_title("選擇的欄位有問題", color="red")
            self.canvas.draw()
            self.update_table()

    def load_excel_file(self):
        """
        打開檔案選擇對話框，讀取 Excel 或 CSV 檔案。
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "選擇 Excel 或 CSV 檔案", "", "支援的檔案 (*.xlsx *.xls *.csv)"
        )
        if filename:
            try:
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext == ".xlsx":
                    engine = "openpyxl"
                    self.excel_data = pd.read_excel(filename, engine=engine)
                elif file_ext == ".xls":
                    engine = "xlrd"
                    self.excel_data = pd.read_excel(filename, engine=engine)
                elif file_ext == ".csv":
                    try:
                        self.excel_data = pd.read_csv(filename, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.excel_data = pd.read_csv(filename, encoding='big5')
                else:
                    QMessageBox.warning(self, "錯誤", "不支援的檔案類型，請選擇 .xlsx、.xls 或 .csv 檔案。")
                    return

                self.x_col_combo.clear()
                self.y_col_combo.clear()
                self.x_col_combo.addItems(self.excel_data.columns)
                self.y_col_combo.addItems(self.excel_data.columns)
                QMessageBox.information(self, "成功", "檔案已載入成功。")
                self.data_source = 'file'
                
                if len(self.excel_data.columns) >= 2:
                    self.x_col_combo.setCurrentIndex(0)
                    self.y_col_combo.setCurrentIndex(1)
                
            except BadZipFile:
                QMessageBox.critical(self, "讀取錯誤", "檔案已損壞或非標準格式。")
            except Exception as e:
                QMessageBox.critical(self, "讀取錯誤", f"無法讀取檔案：{e}")

    def update_plot(self):
        """
        根據當前數據和設定更新繪圖。
        """
        self.ax.clear()
        self.figure.set_facecolor(self.bg_color_hex)
        self.ax.set_facecolor(self.bg_color_hex)
        
        x_to_plot, y_to_plot = self.x_data, self.y_data
        colors_to_plot = self.colors_data

        x_is_numeric = all(isinstance(x, (int, float)) for x in x_to_plot)
        y_is_numeric = all(isinstance(y, (int, float)) for y in y_to_plot)

        if not x_to_plot or not y_to_plot:
            self.ax.set_title("請輸入或選擇數據以繪製圖表")
            self.canvas.draw()
            return

        if len(colors_to_plot) != len(x_to_plot):
            colors_to_plot = [self.plot_color_hex] * len(x_to_plot)
            self.colors_data = colors_to_plot

        if self.line_checkbox.isChecked():
            if x_is_numeric and y_is_numeric and SCIPY_AVAILABLE and self.smooth_line_checkbox.isChecked():
                try:
                    sorted_indices = np.argsort(x_to_plot)
                    sorted_x = np.array(x_to_plot)[sorted_indices]
                    sorted_y = np.array(y_to_plot)[sorted_indices]
                    interpolator = PchipInterpolator(sorted_x, sorted_y)
                    x_smooth = np.linspace(min(sorted_x), max(sorted_x), 300)
                    y_smooth = interpolator(x_smooth)
                    
                    self.ax.plot(x_smooth, y_smooth,
                                 linestyle='-',
                                 color=self.plot_color_hex,
                                 linewidth=self.line_width_spinbox.value(),
                                 zorder=1, label="平滑曲線")
                except Exception as e:
                    print(f"無法生成平滑曲線: {e}")
                    QMessageBox.warning(self, "平滑曲線錯誤", f"無法為當前數據生成平滑曲線，將改為繪製普通折線圖。\n\n錯誤訊息：{e}")
                    linestyle_map = {"實線": "-", "虛線": "--", "點虛線": "-.", "點": ":"}
                    selected_linestyle = linestyle_map.get(self.linestyle_combo.currentText(), '-')
                    self.ax.plot(x_to_plot, y_to_plot,
                                 linestyle=selected_linestyle,
                                 color=self.plot_color_hex,
                                 linewidth=self.line_width_spinbox.value(),
                                 zorder=1, label="折線圖")
            else:
                 linestyle_map = {"實線": "-", "虛線": "--", "點虛線": "-.", "點": ":"}
                 selected_linestyle = linestyle_map.get(self.linestyle_combo.currentText(), '-')
                 self.ax.plot(x_to_plot, y_to_plot,
                                 linestyle=selected_linestyle,
                                 color=self.plot_color_hex,
                                 linewidth=self.line_width_spinbox.value(),
                                 zorder=1, label="折線圖")

        if self.scatter_checkbox.isChecked() or self.line_checkbox.isChecked():
            marker_map = {"圓形": "o", "方形": "s", "三角形": "^", "星形": "*", "無": "None"}
            selected_marker = marker_map.get(self.marker_combo.currentText(), 'o')
            
            if selected_marker != "None":
                self.ax.scatter(x_to_plot, y_to_plot,
                                s=self.point_size_spinbox.value(),
                                marker=selected_marker,
                                c=colors_to_plot,
                                edgecolors=self.border_color_hex,
                                linewidths=self.border_width_spinbox.value(),
                                zorder=2)
        
        if self.bar_checkbox.isChecked():
            self.ax.bar(x_to_plot, y_to_plot,
                        width=self.bar_width_spinbox.value(),
                        color=colors_to_plot,
                        edgecolor=self.border_color_hex,
                        linewidth=self.border_width_spinbox.value(),
                        zorder=2, label="長條圖")

        if self.show_data_labels_checkbox.isChecked():
            for x, y, color in zip(x_to_plot, y_to_plot, colors_to_plot):
                label_parts = []
                if self.show_x_labels_checkbox.isChecked():
                    if isinstance(x, (int, float)):
                        label_parts.append(f"{x:.{self.x_decimal_spinbox.value()}f}")
                    else:
                        label_parts.append(f"{x}")
                if self.show_y_labels_checkbox.isChecked():
                    if isinstance(y, (int, float)):
                        label_parts.append(f"{y:.{self.y_decimal_spinbox.value()}f}")
                    else:
                        label_parts.append(f"{y}")

                label_text = ", ".join(label_parts)
                if label_text:
                    self.ax.annotate(label_text, (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                                     fontsize=self.data_label_size_spinbox.value(), color=color)

        self.ax.set_title(self.title_input.text() or "多功能圖表")
        
        x_label_text = self.x_label_input.text()
        y_label_text = self.y_label_input.text()
        
        self.ax.set_xlabel(x_label_text, color=self.x_label_color_hex,
                           fontweight='bold' if self.x_label_bold_checkbox.isChecked() else 'normal',
                           fontsize=self.x_label_size_spinbox.value())
        self.ax.set_ylabel(y_label_text, color=self.y_label_color_hex,
                           fontweight='bold' if self.y_label_bold_checkbox.isChecked() else 'normal',
                           fontsize=self.y_label_size_spinbox.value())
        
        tick_direction_map = {"朝外": "out", "朝內": "in", "朝內外": "inout"}
        selected_tick_direction = tick_direction_map.get(self.tick_direction_combo.currentText(), "out")

        self.ax.tick_params(axis='x', labelsize=self.x_tick_label_size_spinbox.value(), direction=selected_tick_direction)
        self.ax.tick_params(axis='y', labelsize=self.y_tick_label_size_spinbox.value(), direction=selected_tick_direction)
        
        if self.minor_grid_checkbox.isChecked():
            self.ax.minorticks_on()
            self.ax.tick_params(which='minor', axis='both', length=self.minor_tick_length_spinbox.value(), direction=selected_tick_direction)

        if self.x_interval_spinbox.value() > 0 and x_is_numeric:
            self.ax.xaxis.set_major_locator(ticker.MultipleLocator(self.x_interval_spinbox.value()))
        if self.y_interval_spinbox.value() > 0 and y_is_numeric:
            self.ax.yaxis.set_major_locator(ticker.MultipleLocator(self.y_interval_spinbox.value()))
        
        if self.minor_x_interval_spinbox.value() > 0 and x_is_numeric:
            self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(self.minor_x_interval_spinbox.value()))
        else:
            self.ax.xaxis.set_minor_locator(ticker.NullLocator())

        if self.minor_y_interval_spinbox.value() > 0 and y_is_numeric:
            self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(self.minor_y_interval_spinbox.value()))
        else:
            self.ax.yaxis.set_minor_locator(ticker.NullLocator())

        for spine in self.ax.spines.values():
            spine.set_linewidth(self.axis_border_width_spinbox.value())
            
        self.ax.grid(self.major_grid_checkbox.isChecked(), which='major', color=self.major_grid_color_hex, linestyle='-', linewidth=0.5)
        self.ax.grid(self.minor_grid_checkbox.isChecked(), which='minor', color=self.minor_grid_color_hex, linestyle=':', linewidth=0.5)

        
        handles, labels = self.ax.get_legend_handles_labels()
        legend_label_input = self.legend_input.text()
        
        if self.legend:
            self.legend.remove()
            self.legend = None

        if legend_label_input and handles:
            handles_for_legend = []
            for handle, label in zip(handles, labels):
                if label:
                    handles_for_legend.append(handle)
            
            if handles_for_legend:
                self.legend = self.ax.legend(
                    handles_for_legend[:1],
                    [legend_label_input],
                    prop={'size': self.legend_size_spinbox.value()},
                    draggable=True
                )

        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_table(self):
        """
        根據當前數據更新表格。
        """
        self.is_updating_table = True
        
        row_count = len(self.x_data)
        self.data_table.setRowCount(row_count)
        
        for i in range(row_count):
            x_val = str(self.x_data[i])
            y_val = str(self.y_data[i])
            
            if i < len(self.colors_data):
                color_val = self.colors_data[i]
            else:
                color_val = self.plot_color_hex
                self.colors_data.append(color_val)

            x_item = QTableWidgetItem(x_val)
            y_item = QTableWidgetItem(y_val)
            color_item = QTableWidgetItem(color_val)
            
            color_item.setBackground(QColor(color_val))

            self.data_table.setItem(i, 0, x_item)
            self.data_table.setItem(i, 1, y_item)
            self.data_table.setItem(i, 2, color_item)

        self.is_updating_table = False
            
    def add_row(self):
        """
        在表格中新增一行。
        """
        self.x_data.append(0)
        self.y_data.append(0)
        self.colors_data.append(self.plot_color_hex)
        self.update_table()
        self.update_plot_with_timer()

    def remove_row(self):
        """
        從表格中移除選定的行。
        """
        selected_rows = sorted(list(set(index.row() for index in self.data_table.selectedIndexes())), reverse=True)
        if not selected_rows:
            QMessageBox.warning(self, "警告", "請選擇要刪除的行。")
            return
        
        for row in selected_rows:
            if 0 <= row < len(self.x_data):
                del self.x_data[row]
                del self.y_data[row]
                del self.colors_data[row]
        
        self.update_table()
        self.update_plot_with_timer()

    def move_row_up(self):
        """
        將選定的行上移。
        """
        selected_rows = [index.row() for index in self.data_table.selectedIndexes()]
        if len(selected_rows) != 1 or selected_rows[0] == 0:
            return
        
        row_index = selected_rows[0]
        
        self.x_data[row_index], self.x_data[row_index-1] = self.x_data[row_index-1], self.x_data[row_index]
        self.y_data[row_index], self.y_data[row_index-1] = self.y_data[row_index-1], self.y_data[row_index]
        self.colors_data[row_index], self.colors_data[row_index-1] = self.colors_data[row_index-1], self.colors_data[row_index]

        self.update_table()
        self.update_plot_with_timer()
        self.data_table.selectRow(row_index - 1)

    def move_row_down(self):
        """
        將選定的行下移。
        """
        selected_rows = [index.row() for index in self.data_table.selectedIndexes()]
        if len(selected_rows) != 1 or selected_rows[0] == self.data_table.rowCount() - 1:
            return

        row_index = selected_rows[0]

        self.x_data[row_index], self.x_data[row_index+1] = self.x_data[row_index+1], self.x_data[row_index]
        self.y_data[row_index], self.y_data[row_index+1] = self.y_data[row_index+1], self.y_data[row_index]
        self.colors_data[row_index], self.colors_data[row_index+1] = self.colors_data[row_index+1], self.colors_data[row_index]

        self.update_table()
        self.update_plot_with_timer()
        self.data_table.selectRow(row_index + 1)
        
    def clear_plot(self):
        """
        清除所有數據、輸入框和圖表。
        """
        self.x_data = []
        self.y_data = []
        self.colors_data = []
        self.excel_data = None
        self.data_source = 'manual'
        
        self.title_input.clear()
        self.x_label_input.clear()
        self.y_label_input.clear()
        self.legend_input.clear()
        
        self.x_col_combo.clear()
        self.y_col_combo.clear()
        
        self.update_table()
        self.update_plot()

    def get_settings(self):
        """
        獲取當前所有設定並以字典形式返回。
        """
        return {
            "title": self.title_input.text(),
            "x_label": self.x_label_input.text(),
            "y_label": self.y_label_input.text(),
            "legend": self.legend_input.text(),
            "plot_color_hex": self.plot_color_hex,
            "bg_color_hex": self.bg_color_hex,
            "major_grid_color_hex": self.major_grid_color_hex,
            "minor_grid_color_hex": self.minor_grid_color_hex,
            "border_color_hex": self.border_color_hex,
            "x_label_color_hex": self.x_label_color_hex,
            "y_label_color_hex": self.y_label_color_hex,
            "x_interval": self.x_interval_spinbox.value(),
            "y_interval": self.y_interval_spinbox.value(),
            "x_decimal": self.x_decimal_spinbox.value(),
            "y_decimal": self.y_decimal_spinbox.value(),
            "show_data_labels": self.show_data_labels_checkbox.isChecked(),
            "show_x_labels": self.show_x_labels_checkbox.isChecked(),
            "show_y_labels": self.show_y_labels_checkbox.isChecked(),
            "data_label_size": self.data_label_size_spinbox.value(),
            "is_line_checked": self.line_checkbox.isChecked(),
            "is_scatter_checked": self.scatter_checkbox.isChecked(),
            "is_bar_checked": self.bar_checkbox.isChecked(),
            "line_width": self.line_width_spinbox.value(),
            "bar_width": self.bar_width_spinbox.value(),
            "border_width": self.border_width_spinbox.value(),
            "point_size": self.point_size_spinbox.value(),
            "x_label_size": self.x_label_size_spinbox.value(),
            "y_label_size": self.y_label_size_spinbox.value(),
            "x_label_bold": self.x_label_bold_checkbox.isChecked(),
            "y_label_bold": self.y_label_bold_checkbox.isChecked(),
            "linestyle": self.linestyle_combo.currentText(),
            "marker": self.marker_combo.currentText(),
            "show_major_grid": self.major_grid_checkbox.isChecked(),
            "show_minor_grid": self.minor_grid_checkbox.isChecked(),
            "axis_border_width": self.axis_border_width_spinbox.value(),
            "legend_size": self.legend_size_spinbox.value(),
            "tick_direction": self.tick_direction_combo.currentText(),
            "minor_x_interval": self.minor_x_interval_spinbox.value(),
            "minor_y_interval": self.minor_y_interval_spinbox.value(),
            "minor_tick_length": self.minor_tick_length_spinbox.value(),
            "smooth_line": self.smooth_line_checkbox.isChecked(),
        }

    def set_settings(self, settings):
        """
        根據傳入的字典設定更新 GUI。
        """
        self.title_input.setText(settings.get("title", ""))
        self.x_label_input.setText(settings.get("x_label", ""))
        self.y_label_input.setText(settings.get("y_label", ""))
        self.legend_input.setText(settings.get("legend", ""))
        self.plot_color_hex = settings.get("plot_color_hex", "#1f77b4")
        self.bg_color_hex = settings.get("bg_color_hex", "#ffffff")
        self.major_grid_color_hex = settings.get("major_grid_color_hex", "#cccccc")
        self.minor_grid_color_hex = settings.get("minor_grid_color_hex", "#eeeeee")
        self.border_color_hex = settings.get("border_color_hex", "#000000")
        self.x_label_color_hex = settings.get("x_label_color_hex", "#000000")
        self.y_label_color_hex = settings.get("y_label_color_hex", "#000000")
        self.x_interval_spinbox.setValue(settings.get("x_interval", 1.0))
        self.y_interval_spinbox.setValue(settings.get("y_interval", 1.0))
        self.x_decimal_spinbox.setValue(settings.get("x_decimal", 2))
        self.y_decimal_spinbox.setValue(settings.get("y_decimal", 2))
        self.show_data_labels_checkbox.setChecked(settings.get("show_data_labels", False))
        self.show_x_labels_checkbox.setChecked(settings.get("show_x_labels", True))
        self.show_y_labels_checkbox.setChecked(settings.get("show_y_labels", True))
        self.data_label_size_spinbox.setValue(settings.get("data_label_size", 10))
        self.line_checkbox.setChecked(settings.get("is_line_checked", True))
        self.scatter_checkbox.setChecked(settings.get("is_scatter_checked", False))
        self.bar_checkbox.setChecked(settings.get("is_bar_checked", False))
        self.line_width_spinbox.setValue(settings.get("line_width", 2.0))
        self.bar_width_spinbox.setValue(settings.get("bar_width", 0.8))
        self.border_width_spinbox.setValue(settings.get("border_width", 1.0))
        self.point_size_spinbox.setValue(settings.get("point_size", 10.0))
        self.x_label_size_spinbox.setValue(settings.get("x_label_size", 12))
        self.y_label_size_spinbox.setValue(settings.get("y_label_size", 12))
        self.x_label_bold_checkbox.setChecked(settings.get("x_label_bold", False))
        self.y_label_bold_checkbox.setChecked(settings.get("y_label_bold", False))
        self.legend_size_spinbox.setValue(settings.get("legend_size", 10))
        
        linestyle_index = self.linestyle_combo.findText(settings.get("linestyle", "實線"))
        if linestyle_index != -1:
            self.linestyle_combo.setCurrentIndex(linestyle_index)
            
        marker_index = self.marker_combo.findText(settings.get("marker", "圓形"))
        if marker_index != -1:
            self.marker_combo.setCurrentIndex(marker_index)
            
        self.major_grid_checkbox.setChecked(settings.get("show_major_grid", True))
        self.minor_grid_checkbox.setChecked(settings.get("show_minor_grid", False))
        self.axis_border_width_spinbox.setValue(settings.get("axis_border_width", 1.0))
        
        tick_direction_index = self.tick_direction_combo.findText(settings.get("tick_direction", "朝外"))
        if tick_direction_index != -1:
            self.tick_direction_combo.setCurrentIndex(tick_direction_index)
        
        self.minor_x_interval_spinbox.setValue(settings.get("minor_x_interval", 0.5))
        self.minor_y_interval_spinbox.setValue(settings.get("minor_y_interval", 0.5))
        self.minor_tick_length_spinbox.setValue(settings.get("minor_tick_length", 4.0))
        
        if SCIPY_AVAILABLE:
            self.smooth_line_checkbox.setChecked(settings.get("smooth_line", False))

        self.update_button_color()
        self.toggle_plot_settings()
        self.update_plot_with_timer()
        
    def create_collapsible_container(self, title, content_layout):
        """
        建立可折疊的容器，包含標題按鈕和內容區。
        """
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        title_button = QPushButton(f"▼ {title}")
        title_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 5px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        content_widget.setVisible(False)
        
        title_button.clicked.connect(lambda: content_widget.setVisible(not content_widget.isVisible()))
        
        container_layout.addWidget(title_button)
        container_layout.addWidget(content_widget)

        return container
        

    def save_template(self):
        """
        將當前設定儲存為 JSON 範本檔案。
        """
        filename, _ = QFileDialog.getSaveFileName(self, "儲存範本", "", "JSON 檔案 (*.json)")
        if filename:
            try:
                settings = self.get_settings()
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "成功", "設定範本已儲存。")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法儲存檔案：{e}")

    def load_template(self):
        """
        從 JSON 檔案載入設定範本。
        """
        filename, _ = QFileDialog.getOpenFileName(self, "載入範本", "", "JSON 檔案 (*.json)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.set_settings(settings)
                QMessageBox.information(self, "成功", "設定範本已載入。")
            except json.JSONDecodeError:
                QMessageBox.critical(self, "讀取錯誤", "無效的 JSON 檔案。")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法載入檔案：{e}")
                
if __name__ == '__main__':
    try:
        required_libraries = ['PySide6', 'pandas', 'matplotlib', 'numpy']
        for lib in required_libraries:
            if lib not in sys.modules:
                __import__(lib)
        
        app = QApplication(sys.argv)
        main_window = PlottingApp()
        main_window.show()
        sys.exit(app.exec())
    except ImportError as e:
        QMessageBox.critical(None, "啟動失敗", f"程式啟動失敗，請確認所有必要的函式庫都已安裝。\n\n詳細錯誤：{e}")
    except Exception as e:
        QMessageBox.critical(None, "程式啟動錯誤", f"啟動時發生未預期的錯誤。\n\n詳細錯誤：{e}")

