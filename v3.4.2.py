import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QMessageBox,
    QColorDialog, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QSpinBox, QScrollArea, QSizePolicy, QFrame, QListWidget, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence
import os
from zipfile import BadZipFile
import json
import numpy as np
import multiprocessing  # <--- 新增: 引入多核心處理模組

# 檢查 scipy 是否存在，並處理 ImportError
try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: 找不到 scipy 函式庫，平滑曲線功能將不可用。請使用 'pip install scipy' 進行安裝。")


# ==============================================================================
# <--- 新增: 多核心處理的工作函式 (START) --->
# 必須將此函式定義在最上層，以便子處理程序可以找到它
# ==============================================================================
def calculate_smooth_curve_worker(args):
    """
    為單一數據集計算平滑曲線。
    這個函式將被每個獨立的 CPU 核心執行。
    """
    x_data, y_data, dataset_index = args
    try:
        # 數據必須先排序才能進行插值
        sorted_indices = np.argsort(x_data)
        sorted_x = np.array(x_data)[sorted_indices]
        sorted_y = np.array(y_data)[sorted_indices]

        # PchipInterpolator 需要唯一的 x 值，如果存在重複的 x，則取其 y 值的平均
        unique_x, unique_indices, counts = np.unique(sorted_x, return_index=True, return_counts=True)
        if len(unique_x) < 2:
            # 如果唯一數據點少於2個，無法插值，返回原始數據
            return (dataset_index, sorted_x, sorted_y, None)

        if len(unique_x) < len(sorted_x):
            # 處理重複的 x 值
            final_y = [np.mean(sorted_y[sorted_x == x_val]) for x_val in unique_x]
            final_x, final_y = unique_x, np.array(final_y)
        else:
            final_x, final_y = sorted_x, sorted_y

        # 建立插值器並計算平滑曲線的點
        interpolator = PchipInterpolator(final_x, final_y)
        x_smooth = np.linspace(min(final_x), max(final_x), 300)
        y_smooth = interpolator(x_smooth)

        # 返回計算結果和原始索引
        return (dataset_index, x_smooth, y_smooth, None)
    except Exception as e:
        # 如果發生錯誤，返回錯誤訊息
        return (dataset_index, None, None, str(e))
# ==============================================================================
# <--- 新增: 多核心處理的工作函式 (END) --->
# ==============================================================================


# 輔助類別，用於使 Matplotlib 的 artist (如文字) 可拖曳
class DraggableArtist:
    """ 一個讓 Matplotlib artist (例如 Title, Label) 可被滑鼠拖曳的類別。 """
    def __init__(self, artist, canvas):
        self.artist = artist
        self.canvas = canvas
        self.is_dragging = False
        self.press_offset = (0, 0)
        
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        """ 處理滑鼠按下事件 (使用像素座標以確保縮放穩定性) """
        if event.inaxes is None or self.is_dragging:
            return
        
        contains, _ = self.artist.contains(event)
        if not contains:
            return
            
        self.is_dragging = True
        
        pos_data = self.artist.get_position()
        pos_pixels = self.artist.get_transform().transform(pos_data)

        self.press_offset = (pos_pixels[0] - event.x, pos_pixels[1] - event.y)

    def on_motion(self, event):
        """ 處理滑鼠移動事件 (使用像素座標) """
        if not self.is_dragging or event.inaxes is None or event.x is None:
            return
            
        x_pixel_new = event.x + self.press_offset[0]
        y_pixel_new = event.y + self.press_offset[1]
        
        new_pos_native = self.artist.get_transform().inverted().transform((x_pixel_new, y_pixel_new))

        self.artist.set_position(new_pos_native)
        self.canvas.draw_idle()

    def on_release(self, event):
        """ 處理滑鼠釋放事件 """
        self.is_dragging = False
        
    def disconnect(self):
        """ 斷開所有事件連接，用於清理 """
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_release)


class PlottingApp(QMainWindow):
    """
    主要應用程式視窗類別，包含 GUI 和所有繪圖邏輯。
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("多功能繪圖工具-V3.4.1 (多核心加速版)") # 版本號更新
        self.setGeometry(100, 100, 1200, 800)
        
        self.set_matplotlib_font()
        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.legend = None

        self.excel_data = None
        self.data_source = 'manual'
        self.is_updating_table = False
        self.is_updating_ui = False
        self.original_datasets = []
        self.last_sort_info = {'col': -1, 'count': 0}
        
        self.selected_artist_info = None
        
        self.datasets = []
        self.artists_map = {}

        self.annotations = []
        self.dragged_annotation = None
        self.drag_start_offset = (0, 0)
        self.annotation_positions = {}
        
        self.draggable_handlers = []
        
        self.highlighted_widgets = []

        self.line_color_hex = "#1f77b4"
        self.point_color_hex = "#ff7f0e"
        self.bg_color_hex = "#ffffff"
        self.major_grid_color_hex = "#cccccc"
        self.minor_grid_color_hex = "#eeeeee"
        self.border_color_hex = "#000000"
        self.x_label_color_hex = "#000000"
        self.y_label_color_hex = "#000000"

        self.init_ui()

        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_press_event', self.on_press_annotate)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_annotate)
        self.canvas.mpl_connect('button_release_event', self.on_release_annotate)
        
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()

        self.update_button_color()
        self.update_plot()
        self.update_table()
        self.toggle_plot_settings()
        
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
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def init_ui(self):
        """
        初始化 GUI 介面元件。
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(400)
        
        self.data_settings_tab = QWidget()
        self.plot_settings_tab = QWidget()
        
        self.control_tabs.addTab(self.data_settings_tab, "數據與檔案")
        self.control_tabs.addTab(self.plot_settings_tab, "繪圖設定")

        main_layout.addWidget(self.control_tabs)

        plot_area_widget = QWidget()
        plot_area_layout = QVBoxLayout(plot_area_widget)
        
        self.info_label = QLabel(" ")
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.info_label.setStyleSheet("background-color: rgba(255, 255, 255, 0.7); padding: 2px; border-radius: 3px;")
        
        toolbar_info_layout = QHBoxLayout()
        toolbar_info_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_info_layout.addWidget(self.toolbar)
        toolbar_info_layout.addStretch()
        toolbar_info_layout.addWidget(self.info_label)
        
        plot_area_layout.addLayout(toolbar_info_layout, 0)
        plot_area_layout.addWidget(self.canvas, 1)
        
        main_layout.addWidget(plot_area_widget)

        data_settings_layout = QVBoxLayout(self.data_settings_tab)
        data_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        excel_layout = QGridLayout()
        self.load_excel_btn = QPushButton("選擇 Excel 或 CSV 檔案")
        self.load_excel_btn.clicked.connect(self.load_excel_file)
        excel_layout.addWidget(self.load_excel_btn, 0, 0, 1, 2)
        
        excel_layout.addWidget(QLabel("X 欄位:"), 1, 0)
        self.x_col_combo = QComboBox()
        self.x_col_combo.currentIndexChanged.connect(self.update_data_from_file_input)
        excel_layout.addWidget(self.x_col_combo, 1, 1)
        
        excel_layout.addWidget(QLabel("Y 欄位 (可多選):"), 2, 0, 1, 2)
        self.y_col_list = QListWidget()
        self.y_col_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.y_col_list.itemSelectionChanged.connect(self.update_data_from_file_input)
        excel_layout.addWidget(self.y_col_list, 3, 0, 1, 2)
        excel_group = self.create_collapsible_container("檔案讀取", excel_layout)
        data_settings_layout.addWidget(excel_group)
        
        data_table_layout = QVBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("輸入文字以篩選表格...")
        self.filter_input.textChanged.connect(self.filter_table)
        data_table_layout.addWidget(self.filter_input)
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["X 數據", "Y 數據", "顏色"])
        self.data_table.setColumnWidth(2, 60)
        self.data_table.itemChanged.connect(self.update_data_from_table)
        self.data_table.cellClicked.connect(self.pick_color_for_cell)
        self.data_table.setSortingEnabled(True)
        self.data_table.keyPressEvent = self.table_key_press_event
        self.data_table.horizontalHeader().sortIndicatorChanged.connect(self.on_table_sort)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        data_table_layout.addWidget(self.data_table)

        table_control_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("新增行")
        self.add_row_btn.clicked.connect(self.add_row)
        self.remove_row_btn = QPushButton("刪除行")
        self.remove_row_btn.clicked.connect(self.remove_row)
        self.move_up_btn = QPushButton("上移")
        self.move_down_btn = QPushButton("下移")
        self.move_up_btn.clicked.connect(self.move_row_up)
        self.move_down_btn.clicked.connect(self.move_row_down)
        table_control_layout.addWidget(self.add_row_btn)
        table_control_layout.addWidget(self.remove_row_btn)
        table_control_layout.addWidget(self.move_up_btn)
        table_control_layout.addWidget(self.move_down_btn)
        data_table_layout.addLayout(table_control_layout)
        data_table_group = self.create_collapsible_container("數據表格", data_table_layout)
        data_settings_layout.addWidget(data_table_group)

        self.plot_settings_scroll_area = QScrollArea()
        self.plot_settings_scroll_area.setWidgetResizable(True)
        plot_settings_content_widget = QWidget()
        plot_settings_layout = QVBoxLayout(plot_settings_content_widget)
        plot_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        plot_type_layout = QHBoxLayout()
        self.line_checkbox = QCheckBox("折線圖")
        self.scatter_checkbox = QCheckBox("散佈圖")
        self.bar_checkbox = QCheckBox("長條圖")
        self.box_checkbox = QCheckBox("盒鬚圖")
        self.line_checkbox.toggled.connect(self.toggle_plot_settings)
        self.scatter_checkbox.toggled.connect(self.toggle_plot_settings)
        self.bar_checkbox.toggled.connect(self.toggle_plot_settings)
        self.box_checkbox.toggled.connect(self.toggle_plot_settings)
        plot_type_layout.addWidget(self.line_checkbox)
        plot_type_layout.addWidget(self.scatter_checkbox)
        plot_type_layout.addWidget(self.bar_checkbox)
        plot_type_layout.addWidget(self.box_checkbox)
        plot_type_group = self.create_collapsible_container("圖表類型", plot_type_layout)
        plot_settings_layout.addWidget(plot_type_group)

        self.settings_layout = QVBoxLayout()
        self.settings_layout.addWidget(QLabel("圖表標題:"))
        self.title_input = QLineEdit()
        self.title_input.textChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.title_input)
        self.settings_layout.addWidget(QLabel("X 軸標籤:"))
        self.x_label_input = QLineEdit()
        self.x_label_input.textChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.x_label_input)
        self.settings_layout.addWidget(QLabel("Y 軸標籤:"))
        self.y_label_input = QLineEdit()
        self.y_label_input.textChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.y_label_input)
        self.settings_layout.addWidget(QLabel("圖例文字大小:"))
        self.legend_size_spinbox = QSpinBox()
        self.legend_size_spinbox.setMinimum(1)
        self.legend_size_spinbox.setValue(10)
        self.legend_size_spinbox.valueChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.legend_size_spinbox)
        
        self.settings_layout.addWidget(QLabel("顯示數據值:"))
        data_label_layout = QHBoxLayout()
        self.show_data_labels_checkbox = QCheckBox("顯示數據值")
        self.show_data_labels_checkbox.toggled.connect(self.update_plot)
        data_label_layout.addWidget(self.show_data_labels_checkbox)
        self.show_x_labels_checkbox = QCheckBox("X")
        self.show_x_labels_checkbox.setChecked(True)
        self.show_x_labels_checkbox.toggled.connect(self.update_plot)
        data_label_layout.addWidget(self.show_x_labels_checkbox)
        self.show_y_labels_checkbox = QCheckBox("Y")
        self.show_y_labels_checkbox.setChecked(True)
        self.show_y_labels_checkbox.toggled.connect(self.update_plot)
        data_label_layout.addWidget(self.show_y_labels_checkbox)
        self.settings_layout.addLayout(data_label_layout)
        
        self.settings_layout.addWidget(QLabel("數據標籤大小:"))
        self.data_label_size_spinbox = QSpinBox()
        self.data_label_size_spinbox.setMinimum(1)
        self.data_label_size_spinbox.setValue(10)
        self.data_label_size_spinbox.valueChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.data_label_size_spinbox)
        
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("X軸間隔:"))
        self.x_interval_spinbox = QDoubleSpinBox()
        self.x_interval_spinbox.setMinimum(0.0)
        self.x_interval_spinbox.setMaximum(float('inf'))
        self.x_interval_spinbox.setValue(1.0)
        self.x_interval_spinbox.valueChanged.connect(self.update_plot)
        interval_layout.addWidget(self.x_interval_spinbox)
        self.settings_layout.addLayout(interval_layout)

        interval_layout_y = QHBoxLayout()
        interval_layout_y.addWidget(QLabel("Y軸間隔:"))
        self.y_interval_spinbox = QDoubleSpinBox()
        self.y_interval_spinbox.setMinimum(0.0)
        self.y_interval_spinbox.setMaximum(float('inf'))
        self.y_interval_spinbox.setValue(1.0)
        self.y_interval_spinbox.valueChanged.connect(self.update_plot)
        interval_layout_y.addWidget(self.y_interval_spinbox)
        self.settings_layout.addLayout(interval_layout_y)

        decimal_layout = QHBoxLayout()
        decimal_layout.addWidget(QLabel("X軸小數點:"))
        self.x_decimal_spinbox = QSpinBox()
        self.x_decimal_spinbox.setValue(2)
        self.x_decimal_spinbox.valueChanged.connect(self.update_plot)
        decimal_layout.addWidget(self.x_decimal_spinbox)
        self.settings_layout.addLayout(decimal_layout)
        
        decimal_layout_y = QHBoxLayout()
        decimal_layout_y.addWidget(QLabel("Y軸小數點:"))
        self.y_decimal_spinbox = QSpinBox()
        self.y_decimal_spinbox.setValue(2)
        self.y_decimal_spinbox.valueChanged.connect(self.update_plot)
        decimal_layout_y.addWidget(self.y_decimal_spinbox)
        self.settings_layout.addLayout(decimal_layout_y)

        self.settings_group = self.create_collapsible_container("圖表設定", self.settings_layout, "settings_group")
        plot_settings_layout.addWidget(self.settings_group)
        
        self.style_layout = QVBoxLayout()
        self.line_width_spinbox = QDoubleSpinBox()
        self.bar_width_spinbox = QDoubleSpinBox()
        self.point_size_spinbox = QDoubleSpinBox()
        self.linestyle_combo = QComboBox()
        self.marker_combo = QComboBox()
        self.connect_scatter_checkbox = QCheckBox("連接散佈點")
        self.connect_scatter_checkbox.toggled.connect(self.update_plot)
        self.smooth_line_checkbox = QCheckBox("平滑曲線")
        self.smooth_line_checkbox.toggled.connect(self.update_plot)
        
        series_selection_layout = QHBoxLayout()
        series_selection_layout.addWidget(QLabel("選擇數據系列:"))
        self.series_combo = QComboBox()
        self.series_combo.currentIndexChanged.connect(self.on_series_selected_from_combo)
        series_selection_layout.addWidget(self.series_combo)
        self.style_layout.addLayout(series_selection_layout)

        self.setup_dynamic_widgets()
        
        line_color_layout = QHBoxLayout()
        line_color_layout.addWidget(QLabel("線條顏色:"))
        self.line_color_btn = QPushButton("選擇顏色")
        self.line_color_btn.clicked.connect(lambda: self.pick_color("line"))
        line_color_layout.addWidget(self.line_color_btn)
        self.style_layout.addLayout(line_color_layout)

        point_color_layout = QHBoxLayout()
        point_color_layout.addWidget(QLabel("數據點顏色:"))
        self.point_color_btn = QPushButton("選擇顏色")
        self.point_color_btn.clicked.connect(lambda: self.pick_color("point"))
        point_color_layout.addWidget(self.point_color_btn)
        self.style_layout.addLayout(point_color_layout)

        border_width_layout = QHBoxLayout()
        border_width_layout.addWidget(QLabel("邊框粗度:"))
        self.border_width_spinbox = QDoubleSpinBox()
        self.border_width_spinbox.setMinimum(0.0)
        self.border_width_spinbox.setValue(1.0)
        self.border_width_spinbox.setSingleStep(0.5)
        self.border_width_spinbox.valueChanged.connect(self.update_plot)
        border_width_layout.addWidget(self.border_width_spinbox)
        self.style_layout.addLayout(border_width_layout)

        border_color_layout = QHBoxLayout()
        border_color_layout.addWidget(QLabel("邊框顏色:"))
        self.border_color_btn = QPushButton("選擇顏色")
        self.border_color_btn.clicked.connect(lambda: self.pick_color("border"))
        border_color_layout.addWidget(self.border_color_btn)
        self.style_layout.addLayout(border_color_layout)

        self.style_group = self.create_collapsible_container("繪圖樣式設定", self.style_layout, "style_group")
        plot_settings_layout.addWidget(self.style_group)
        
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
        self.x_label_size_spinbox.valueChanged.connect(self.update_plot)
        x_label_props_layout.addWidget(self.x_label_size_spinbox)
        self.x_label_bold_checkbox = QCheckBox("粗體")
        self.x_label_bold_checkbox.toggled.connect(self.update_plot)
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
        self.y_label_size_spinbox.valueChanged.connect(self.update_plot)
        y_label_props_layout.addWidget(self.y_label_size_spinbox)
        self.y_label_bold_checkbox = QCheckBox("粗體")
        self.y_label_bold_checkbox.toggled.connect(self.update_plot)
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
        self.x_tick_label_size_spinbox.valueChanged.connect(self.update_plot)
        tick_size_layout.addWidget(self.x_tick_label_size_spinbox)
        self.x_tick_label_bold_checkbox = QCheckBox("粗體")
        self.x_tick_label_bold_checkbox.toggled.connect(self.update_plot)
        tick_size_layout.addWidget(self.x_tick_label_bold_checkbox)
        axis_style_layout.addLayout(tick_size_layout)

        tick_size_layout_y = QHBoxLayout()
        tick_size_layout_y.addWidget(QLabel("Y軸刻度大小:"))
        self.y_tick_label_size_spinbox = QSpinBox()
        self.y_tick_label_size_spinbox.setMinimum(1)
        self.y_tick_label_size_spinbox.setValue(10)
        self.y_tick_label_size_spinbox.valueChanged.connect(self.update_plot)
        tick_size_layout_y.addWidget(self.y_tick_label_size_spinbox)
        self.y_tick_label_bold_checkbox = QCheckBox("粗體")
        self.y_tick_label_bold_checkbox.toggled.connect(self.update_plot)
        tick_size_layout_y.addWidget(self.y_tick_label_bold_checkbox)
        axis_style_layout.addLayout(tick_size_layout_y)

        tick_rotation_layout_x = QHBoxLayout()
        tick_rotation_layout_x.addWidget(QLabel("X軸刻度角度:"))
        self.x_tick_rotation_spinbox = QSpinBox()
        self.x_tick_rotation_spinbox.setRange(-90, 90)
        self.x_tick_rotation_spinbox.setValue(0)
        self.x_tick_rotation_spinbox.valueChanged.connect(self.update_plot)
        tick_rotation_layout_x.addWidget(self.x_tick_rotation_spinbox)
        axis_style_layout.addLayout(tick_rotation_layout_x)

        tick_rotation_layout_y = QHBoxLayout()
        tick_rotation_layout_y.addWidget(QLabel("Y軸刻度角度:"))
        self.y_tick_rotation_spinbox = QSpinBox()
        self.y_tick_rotation_spinbox.setRange(-90, 90)
        self.y_tick_rotation_spinbox.setValue(0)
        self.y_tick_rotation_spinbox.valueChanged.connect(self.update_plot)
        tick_rotation_layout_y.addWidget(self.y_tick_rotation_spinbox)
        axis_style_layout.addLayout(tick_rotation_layout_y)
        
        axis_style_layout.addWidget(QLabel("座標軸邊框粗度:"))
        self.axis_border_width_spinbox = QDoubleSpinBox()
        self.axis_border_width_spinbox.setMinimum(0.0)
        self.axis_border_width_spinbox.setValue(1.0)
        self.axis_border_width_spinbox.setSingleStep(0.5)
        self.axis_border_width_spinbox.valueChanged.connect(self.update_plot)
        axis_style_layout.addWidget(self.axis_border_width_spinbox)

        self.tick_direction_layout = QHBoxLayout()
        self.tick_direction_layout.addWidget(QLabel("刻度線朝向:"))
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(["朝外", "朝內", "朝內外"])
        self.tick_direction_combo.currentIndexChanged.connect(self.update_plot)
        self.tick_direction_layout.addWidget(self.tick_direction_combo)
        axis_style_layout.addLayout(self.tick_direction_layout)

        major_tick_size_layout = QGridLayout()
        major_tick_size_layout.addWidget(QLabel("主刻度線 長度:"), 0, 0)
        self.major_tick_length_spinbox = QDoubleSpinBox()
        self.major_tick_length_spinbox.setValue(3.5)
        self.major_tick_length_spinbox.valueChanged.connect(self.update_plot)
        major_tick_size_layout.addWidget(self.major_tick_length_spinbox, 0, 1)

        major_tick_size_layout.addWidget(QLabel("寬度:"), 0, 2)
        self.major_tick_width_spinbox = QDoubleSpinBox()
        self.major_tick_width_spinbox.setValue(0.8)
        self.major_tick_width_spinbox.valueChanged.connect(self.update_plot)
        major_tick_size_layout.addWidget(self.major_tick_width_spinbox, 0, 3)
        axis_style_layout.addLayout(major_tick_size_layout)
        
        self.axis_style_group = self.create_collapsible_container("背景與座標軸設定", axis_style_layout, "axis_style_group")
        plot_settings_layout.addWidget(self.axis_style_group)
        
        grid_layout = QVBoxLayout()
        self.major_grid_checkbox = QCheckBox("顯示主網格線")
        self.major_grid_checkbox.setChecked(True)
        self.major_grid_checkbox.toggled.connect(self.update_plot)
        grid_layout.addWidget(self.major_grid_checkbox)
        self.major_grid_color_btn = QPushButton("主網格顏色")
        self.major_grid_color_btn.clicked.connect(lambda: self.pick_color("major_grid"))
        grid_layout.addWidget(self.major_grid_color_btn)
        self.minor_grid_checkbox = QCheckBox("顯示次網格線")
        self.minor_grid_checkbox.toggled.connect(self.update_plot)
        grid_layout.addWidget(self.minor_grid_checkbox)
        self.minor_grid_color_btn = QPushButton("次網格顏色")
        self.minor_grid_color_btn.clicked.connect(lambda: self.pick_color("minor_grid"))
        grid_layout.addWidget(self.minor_grid_color_btn)
        self.grid_group = self.create_collapsible_container("網格線設定", grid_layout, "grid_group")
        plot_settings_layout.addWidget(self.grid_group)
        
        minor_tick_layout = QVBoxLayout()
        minor_tick_layout.addWidget(QLabel("次刻度線 X 軸間隔:"))
        self.minor_x_interval_spinbox = QDoubleSpinBox()
        self.minor_x_interval_spinbox.setValue(0.5)
        self.minor_x_interval_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_layout.addWidget(self.minor_x_interval_spinbox)
        minor_tick_layout.addWidget(QLabel("次刻度線 Y 軸間隔:"))
        self.minor_y_interval_spinbox = QDoubleSpinBox()
        self.minor_y_interval_spinbox.setValue(0.5)
        self.minor_y_interval_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_layout.addWidget(self.minor_y_interval_spinbox)

        minor_tick_size_layout = QGridLayout()
        minor_tick_size_layout.addWidget(QLabel("次刻度線 長度:"), 0, 0)
        self.minor_tick_length_spinbox = QDoubleSpinBox()
        self.minor_tick_length_spinbox.setValue(2.0)
        self.minor_tick_length_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_size_layout.addWidget(self.minor_tick_length_spinbox, 0, 1)
        minor_tick_size_layout.addWidget(QLabel("寬度:"), 0, 2)
        self.minor_tick_width_spinbox = QDoubleSpinBox()
        self.minor_tick_width_spinbox.setValue(0.6)
        self.minor_tick_width_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_size_layout.addWidget(self.minor_tick_width_spinbox, 0, 3)
        minor_tick_layout.addLayout(minor_tick_size_layout)
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

        self.plot_settings_scroll_area.setWidget(plot_settings_content_widget)
        main_plot_settings_layout = QVBoxLayout(self.plot_settings_tab)
        main_plot_settings_layout.addWidget(self.plot_settings_scroll_area)

    def update_series_combo(self):
        """ 更新數據系列選擇下拉選單的內容 """
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        self.series_combo.addItems([ds['name'] for ds in self.datasets])
        
        if self.selected_artist_info:
            ds_index = self.selected_artist_info.get('dataset_index')
            if ds_index is not None and 0 <= ds_index < self.series_combo.count():
                self.series_combo.setCurrentIndex(ds_index)
        
        self.series_combo.blockSignals(False)

    def on_series_selected_from_combo(self, index):
        """ 當使用者從下拉選單選擇一個數據系列時觸發 """
        if index != -1 and not self.is_updating_ui:
            self.select_dataset(index)

    def select_dataset(self, ds_idx):
        """ 根據索引值，以程式化方式選取一個數據系列並更新UI """
        if 0 <= ds_idx < len(self.datasets):
            try:
                self.is_updating_ui = True
                self.selected_artist_info = {'dataset_index': ds_idx, 'point_index': 0}
                current_ds = self.datasets[ds_idx]
                
                if (idx := self.marker_combo.findText(current_ds.get('marker', '無'))) != -1: self.marker_combo.setCurrentIndex(idx)
                if (idx := self.linestyle_combo.findText(current_ds.get('linestyle', '實線'))) != -1: self.linestyle_combo.setCurrentIndex(idx)
                self.line_width_spinbox.setValue(current_ds.get('linewidth', 2.0))
                self.line_color_hex = current_ds.get('primary_color', '#1f77b4')
                self.point_color_hex = current_ds.get('colors', [self.point_color_hex])[0]
                self.border_color_hex = current_ds.get('border_color', '#000000')
                
                self.update_button_color()
                
                self.control_tabs.setCurrentWidget(self.plot_settings_tab)
                self.highlight_widget(self.style_group)
                self.plot_settings_scroll_area.ensureWidgetVisible(self.style_group)
                
            finally:
                self.is_updating_ui = False
        
    def setup_dynamic_widgets(self):
        """ 為動態顯示的組件建立並添加布局。 """
        self.style_layout.addWidget(self.connect_scatter_checkbox)
        
        self.line_width_label = QLabel("線條寬度:")
        self.style_layout.addWidget(self.line_width_label)
        self.line_width_spinbox.setMinimum(0.5)
        self.line_width_spinbox.setValue(2.0)
        self.line_width_spinbox.valueChanged.connect(self.update_artist_style)
        self.style_layout.addWidget(self.line_width_spinbox)

        self.bar_width_label = QLabel("長條圖寬度:")
        self.style_layout.addWidget(self.bar_width_label)
        self.bar_width_spinbox.setMinimum(0.01)
        self.bar_width_spinbox.setValue(0.8)
        self.bar_width_spinbox.setSingleStep(0.05)
        self.bar_width_spinbox.valueChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.bar_width_spinbox)

        self.point_size_label = QLabel("點的大小:")
        self.style_layout.addWidget(self.point_size_label)
        self.point_size_spinbox.setMinimum(1.0)
        self.point_size_spinbox.setValue(10.0)
        self.point_size_spinbox.valueChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.point_size_spinbox)
        
        self.linestyle_label = QLabel("線條樣式:")
        self.style_layout.addWidget(self.linestyle_label)
        self.linestyle_combo.addItems(["實線", "虛線", "點虛線", "點"])
        self.linestyle_combo.currentIndexChanged.connect(self.update_artist_style)
        self.style_layout.addWidget(self.linestyle_combo)
        
        self.marker_label = QLabel("標記樣式:")
        self.style_layout.addWidget(self.marker_label)
        self.marker_combo.addItems(["圓形", "方形", "三角形", "星形", "無"])
        self.marker_combo.currentIndexChanged.connect(self.update_artist_style)
        self.style_layout.addWidget(self.marker_combo)
        
        if SCIPY_AVAILABLE:
            self.style_layout.addWidget(self.smooth_line_checkbox)
    
    def update_artist_style(self):
        """更新選定 artist 的樣式，或在未選定时更新所有 artist 的樣式。"""
        if self.is_updating_ui:
            return
        
        new_marker = self.marker_combo.currentText()
        new_linewidth = self.line_width_spinbox.value()
        new_linestyle = self.linestyle_combo.currentText()

        if self.selected_artist_info:
            ds_index = self.selected_artist_info['dataset_index']
            if 0 <= ds_index < len(self.datasets):
                self.datasets[ds_index]['marker'] = new_marker
                self.datasets[ds_index]['linewidth'] = new_linewidth
                self.datasets[ds_index]['linestyle'] = new_linestyle
        else:
            for dataset in self.datasets:
                dataset['marker'] = new_marker
                dataset['linewidth'] = new_linewidth
                dataset['linestyle'] = new_linestyle

        self.update_plot()

    def update_plot(self):
        """ 根據當前數據和設定更新繪圖。 """
        if self.is_updating_ui:
            return
            
        for handler in self.draggable_handlers:
            handler.disconnect()
        self.draggable_handlers.clear()
        
        self.ax.clear()
        self.artists_map = {}
        self.figure.set_facecolor(self.bg_color_hex)
        self.ax.set_facecolor(self.bg_color_hex)
        
        legend_handles = []
        legend_labels = []

        if not self.datasets or (self.data_source == 'file' and not self.y_col_list.selectedItems()):
            self.ax.set_title("請輸入或選擇數據以繪製圖表")
            self.canvas.draw()
            return

        all_x_numeric = True
        if self.datasets and self.datasets[0]['x']:
             all_x_numeric = all(isinstance(x, (int, float)) for x in self.datasets[0]['x'])
        
        # ==============================================================================
        # <--- 修改: 為平滑曲線準備數據 (START) --->
        # ==============================================================================
        # 建立一個字典來儲存平滑曲線的計算結果
        smooth_results = {}
        # 檢查是否需要進行平滑計算
        should_smooth = self.line_checkbox.isChecked() and SCIPY_AVAILABLE and self.smooth_line_checkbox.isChecked()
        
        if should_smooth:
            # 收集所有需要進行平滑計算的任務
            tasks = []
            for i, dataset in enumerate(self.datasets):
                x_data, y_data = dataset['x'], dataset['y']
                y_is_numeric = all(isinstance(y, (int, float)) for y in y_data)
                x_is_numeric = all(isinstance(x, (int, float)) for x in x_data)
                # 只有 X 和 Y 都是數值類型時才能平滑
                if x_is_numeric and y_is_numeric:
                    tasks.append((x_data, y_data, i))
            
            if tasks:
                try:
                    # 使用 multiprocessing.Pool 來並行執行計算
                    # pool 會自動根據 CPU 核心數建立對應的處理程序
                    with multiprocessing.Pool() as pool:
                        results = pool.map(calculate_smooth_curve_worker, tasks)
                    
                    # 處理計算結果
                    for res_idx, x_smooth, y_smooth, error in results:
                        if error:
                            print(f"數據集 {self.datasets[res_idx]['name']} 無法生成平滑曲線: {error}")
                        else:
                            # 將成功的結果存入字典
                            smooth_results[res_idx] = (x_smooth, y_smooth)
                except Exception as e:
                    print(f"多核心處理平滑曲線時發生錯誤: {e}")
        # ==============================================================================
        # <--- 修改: 為平滑曲線準備數據 (END) --->
        # ==============================================================================

        for dataset_index, dataset in enumerate(self.datasets):
            x_data, y_data, colors, name = dataset['x'], dataset['y'], dataset['colors'], dataset['name']
            
            primary_color = dataset.get('primary_color', self.line_color_hex)
            line_segment_colors = dataset.get('line_segment_colors', [primary_color] * (len(x_data) - 1))
            border_color = dataset.get('border_color', self.border_color_hex)
            marker_style = dataset.get('marker', self.marker_combo.currentText())
            linewidth = dataset.get('linewidth', self.line_width_spinbox.value())
            linestyle_text = dataset.get('linestyle', self.linestyle_combo.currentText())
            linestyle_map = {"實線": "-", "虛線": "--", "點虛線": "-.", "點": ":"}
            ls = linestyle_map.get(linestyle_text, '-')

            y_is_numeric = all(isinstance(y, (int, float)) for y in y_data)
            x_is_numeric = all(isinstance(x, (int, float)) for x in x_data)
            
            if not y_data: continue

            x_plot_data = x_data if x_is_numeric else range(len(x_data))

            # 修改: 繪製圖表部分
            if self.line_checkbox.isChecked() or (self.scatter_checkbox.isChecked() and self.connect_scatter_checkbox.isChecked()):
                # 檢查是否有預先計算好的平滑曲線結果
                if dataset_index in smooth_results:
                    x_smooth, y_smooth = smooth_results[dataset_index]
                    line_artist, = self.ax.plot(x_smooth, y_smooth, linestyle='-', color=primary_color,
                                linewidth=linewidth, zorder=1, label=f"{name} (平滑曲線)")
                    self.artists_map[line_artist] = {'dataset_index': dataset_index, 'type': 'line'}
                    legend_handles.append(line_artist)
                    legend_labels.append(name)
                elif y_is_numeric: # 如果不平滑，則繪製原始折線
                    points = np.array([x_plot_data, y_data]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    if len(line_segment_colors) != len(segments):
                        line_segment_colors = [primary_color] * len(segments)
                        dataset['line_segment_colors'] = line_segment_colors

                    lc = LineCollection(segments, colors=line_segment_colors, linewidths=linewidth, linestyle=ls, zorder=1)
                    line_artist = self.ax.add_collection(lc)
                    
                    proxy_line, = self.ax.plot([], [], color=primary_color, linestyle=ls, label=name)
                    self.artists_map[line_artist] = {'dataset_index': dataset_index, 'type': 'line', 'proxy_artist': proxy_line}
                    legend_handles.append(proxy_line)
                    legend_labels.append(name)

            # ... (原始程式碼的其餘部分保持不變) ...
            if self.scatter_checkbox.isChecked() or (self.line_checkbox.isChecked() and marker_style != "無"):
                if y_is_numeric:
                    marker_map = {"圓形": "o", "方形": "s", "三角形": "^", "星形": "*", "無": "None"}
                    marker = marker_map.get(marker_style, 'o')
                    if marker != "None":
                        scatter = self.ax.scatter(x_plot_data, y_data, s=self.point_size_spinbox.value()**2, marker=marker,
                                                    c=colors, edgecolors=border_color,
                                                    linewidths=self.border_width_spinbox.value(), zorder=2)
                        self.artists_map[scatter] = {'dataset_index': dataset_index, 'type': 'scatter'}
                        if not self.line_checkbox.isChecked():
                            legend_handles.append(scatter)
                            legend_labels.append(name)

            if self.bar_checkbox.isChecked():
                if y_is_numeric:
                    bars = self.ax.bar(x_plot_data, y_data, width=self.bar_width_spinbox.value(), 
                                       color=colors, edgecolor=border_color, linewidth=self.border_width_spinbox.value(), zorder=2, label=name)
                    for rect in bars: self.artists_map[rect] = {'dataset_index': dataset_index, 'type': 'bar'}
                    legend_handles.append(bars[0])
                    legend_labels.append(name)
                else:
                    print("長條圖需要數值型Y軸數據。")
                        
            if self.box_checkbox.isChecked():
                if y_is_numeric:
                    box_plot = self.ax.boxplot(y_data, patch_artist=True, positions=[dataset_index + 1])
                    for patch in box_plot['boxes']: patch.set_facecolor(primary_color)
                    self.ax.set_xticks(range(1, len(self.datasets) + 1))
                    self.ax.set_xticklabels([ds['name'] for ds in self.datasets])
                else:
                    print("盒鬚圖需要數值型數據。")
                
            if self.show_data_labels_checkbox.isChecked():
                if not self.box_checkbox.isChecked():
                    for i, (x, y) in enumerate(zip(x_data, y_data)):
                        label_parts = []
                        if self.show_x_labels_checkbox.isChecked(): 
                            label_parts.append(f"{x:.{self.x_decimal_spinbox.value()}f}" if isinstance(x, (int, float)) else f"{x}")
                        if self.show_y_labels_checkbox.isChecked(): 
                            label_parts.append(f"{y:.{self.y_decimal_spinbox.value()}f}" if isinstance(y, (int, float)) else f"{y}")
                        label_text = ", ".join(label_parts)
                        if label_text:
                            my_id = (dataset_index, i)
                            annot_x = x_plot_data[i]
                            if my_id in self.annotation_positions:
                                pos = self.annotation_positions[my_id]
                                annot = self.ax.annotate(label_text, (annot_x, y), xytext=pos, textcoords='data', ha='center',
                                                            fontsize=self.data_label_size_spinbox.value(), color=colors[i])
                            else:
                                annot = self.ax.annotate(label_text, (annot_x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                                                            fontsize=self.data_label_size_spinbox.value(), color=colors[i])
                            annot.my_id = my_id
                            self.annotations.append(annot)
                elif y_is_numeric:
                    median_val = np.median(y_data)
                    annot = self.ax.annotate(f"中位數: {median_val:.{self.y_decimal_spinbox.value()}f}", 
                                                xy=(dataset_index + 1, median_val),
                                                xytext=(10, 0), textcoords='offset points',
                                                fontsize=self.data_label_size_spinbox.value(), color='black')
                    self.annotations.append(annot)

        # ... (此處省略了約 900 行完全沒有變動的原始程式碼，以節省篇幅) ...
        # ... (從 set_title 到檔案結尾的程式碼都與您提供的原始檔完全相同) ...
        title_obj = self.ax.set_title(self.title_input.text() or "多功能圖表")
        
        if not self.box_checkbox.isChecked():
            xlabel_obj = self.ax.set_xlabel(self.x_label_input.text(), color=self.x_label_color_hex,
                                            weight='bold' if self.x_label_bold_checkbox.isChecked() else 'normal',
                                            fontsize=self.x_label_size_spinbox.value())
        else:
            xlabel_obj = self.ax.set_xlabel("")

        ylabel_obj = self.ax.set_ylabel(self.y_label_input.text(), color=self.y_label_color_hex,
                                        weight='bold' if self.y_label_bold_checkbox.isChecked() else 'normal',
                                        fontsize=self.y_label_size_spinbox.value())
        
        self.draggable_handlers.append(DraggableArtist(title_obj, self.canvas))
        self.draggable_handlers.append(DraggableArtist(xlabel_obj, self.canvas))
        self.draggable_handlers.append(DraggableArtist(ylabel_obj, self.canvas))

        tick_direction_map = {"朝外": "out", "朝內": "in", "朝內外": "inout"}
        tick_dir = tick_direction_map.get(self.tick_direction_combo.currentText(), "out")

        self.ax.tick_params(axis='both', which='major', direction=tick_dir,
                                length=self.major_tick_length_spinbox.value(),
                                width=self.major_tick_width_spinbox.value())
        
        self.ax.tick_params(axis='x', labelsize=self.x_tick_label_size_spinbox.value())
        self.ax.tick_params(axis='y', labelsize=self.y_tick_label_size_spinbox.value())

        plt.setp(self.ax.get_xticklabels(), 
                 fontweight='bold' if self.x_tick_label_bold_checkbox.isChecked() else 'normal',
                 rotation=self.x_tick_rotation_spinbox.value())
        plt.setp(self.ax.get_yticklabels(), 
                 fontweight='bold' if self.y_tick_label_bold_checkbox.isChecked() else 'normal',
                 rotation=self.y_tick_rotation_spinbox.value())
        
        all_y_numeric = all(isinstance(y, (int, float)) for ds in self.datasets for y in ds['y'])
        
        if all_x_numeric:
            if self.x_interval_spinbox.value() > 0 and not self.box_checkbox.isChecked(): self.ax.xaxis.set_major_locator(ticker.MultipleLocator(self.x_interval_spinbox.value()))
            if self.minor_x_interval_spinbox.value() > 0 and not self.box_checkbox.isChecked(): self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(self.minor_x_interval_spinbox.value()))
        elif self.datasets and not self.box_checkbox.isChecked():
            interval = max(1, int(self.x_interval_spinbox.value()))
            x_labels = self.datasets[0]['x']
            
            tick_positions = range(0, len(x_labels), interval)
            tick_labels = [x_labels[i] for i in tick_positions]
            
            self.ax.set_xticks(tick_positions, tick_labels)

        if all_y_numeric:
            if self.y_interval_spinbox.value() > 0: self.ax.yaxis.set_major_locator(ticker.MultipleLocator(self.y_interval_spinbox.value()))
            if self.minor_y_interval_spinbox.value() > 0: self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(self.minor_y_interval_spinbox.value()))

        for spine in self.ax.spines.values(): spine.set_linewidth(self.axis_border_width_spinbox.value())
        
        self.ax.grid(False, which='both')
        if self.major_grid_checkbox.isChecked():
            self.ax.grid(True, which='major', color=self.major_grid_color_hex, linestyle='-', linewidth=0.5)
        
        if self.minor_grid_checkbox.isChecked():
            self.ax.minorticks_on()
            self.ax.grid(True, which='minor', color=self.minor_grid_color_hex, linestyle=':', linewidth=0.5)
            self.ax.tick_params(axis='both', which='minor', length=self.minor_tick_length_spinbox.value(), width=self.minor_tick_width_spinbox.value())
        else:
            self.ax.minorticks_off()

        if legend_handles and legend_labels:
            self.legend = self.ax.legend(handles=legend_handles, labels=legend_labels, prop={'size': self.legend_size_spinbox.value()}, draggable=True)
        else:
            self.legend = None

        try:
            self.figure.tight_layout()
        except ValueError as e:
            print(f"無法自動調整佈局: {e}")
        self.canvas.draw()
        
    def update_table(self):
        self.is_updating_table = True
        if not self.datasets:
            self.data_table.clearContents()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(3)
            self.data_table.setHorizontalHeaderLabels(["X 數據", "Y 數據", "顏色"])
        else:
            num_datasets = len(self.datasets)
            self.data_table.setColumnCount(1 + num_datasets * 2)
            headers = ["X 數據"] + [h for ds in self.datasets for h in (f"Y: {ds['name']}", "顏色")]
            self.data_table.setHorizontalHeaderLabels(headers)
            
            max_rows = max(len(ds.get('x', [])) for ds in self.datasets) if self.datasets else 0
            self.data_table.setRowCount(max_rows)
            
            for row in range(max_rows):
                for i, ds in enumerate(self.datasets):
                    if row < len(ds['x']):
                        if i == 0: self.data_table.setItem(row, 0, QTableWidgetItem(str(ds['x'][row])))
                        y_col, color_col = 1 + i * 2, 2 + i * 2
                        self.data_table.setItem(row, y_col, QTableWidgetItem(str(ds['y'][row])))
                        color_item = QTableWidgetItem('')
                        color_item.setBackground(QColor(ds['colors'][row]))
                        color_item.setFlags(color_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.data_table.setItem(row, color_col, color_item)
        self.is_updating_table = False
            
    def toggle_plot_settings(self):
        is_line = self.line_checkbox.isChecked()
        is_scatter = self.scatter_checkbox.isChecked()
        is_bar = self.bar_checkbox.isChecked()
        
        self.connect_scatter_checkbox.setVisible(is_scatter and not is_line)
        is_line_visible = is_line or (is_scatter and self.connect_scatter_checkbox.isChecked())
        
        for widget in [self.line_width_spinbox, self.line_width_label, self.linestyle_combo, self.linestyle_label, self.smooth_line_checkbox]:
            widget.setVisible(is_line_visible)
        
        for widget in [self.point_size_spinbox, self.point_size_label, self.marker_combo, self.marker_label]:
            widget.setVisible(is_scatter or is_line)
            
        for widget in [self.bar_width_spinbox, self.bar_width_label]:
            widget.setVisible(is_bar)

        self.update_plot()

    def on_canvas_click(self, event):
        self.selected_artist_info = None
        self.clear_all_highlights()

        for annotation in self.annotations:
            if hasattr(annotation, 'is_temp_label'):
                annotation.remove()
        self.annotations = [a for a in self.annotations if not hasattr(a, 'is_temp_label')]
        self.canvas.draw_idle()

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        for handler in self.draggable_handlers:
            if handler.artist.contains(event)[0]:
                self.control_tabs.setCurrentWidget(self.plot_settings_tab)
                self.highlight_widget(self.settings_group)
                if (text_obj := handler.artist) == self.ax.get_title(): self.title_input.setFocus()
                elif text_obj == self.ax.get_xlabel(): self.x_label_input.setFocus()
                elif text_obj == self.ax.get_ylabel(): self.y_label_input.setFocus()
                return

        if (legend := self.ax.get_legend()) and legend.get_visible() and legend.contains(event)[0]:
            self.control_tabs.setCurrentWidget(self.plot_settings_tab)
            self.highlight_widget(self.settings_group)
            self.legend_size_spinbox.setFocus()
            return
        
        pixel_bounds = self.ax.transAxes.transform([(0, 0), (1, 1)])
        x_min_p, y_min_p = pixel_bounds[0]
        x_max_p, y_max_p = pixel_bounds[1]
        
        on_y = (abs(event.x - x_min_p) < 10 or abs(event.x - x_max_p) < 10) and (y_min_p <= event.y <= y_max_p)
        on_x = (abs(event.y - y_min_p) < 10 or abs(event.y - y_max_p) < 10) and (x_min_p <= event.x <= x_max_p)

        if on_x or on_y:
            if not (content := self.axis_style_group.findChild(QWidget)).isVisible(): content.parent().findChild(QPushButton).click()
            self.control_tabs.setCurrentWidget(self.plot_settings_tab)
            self.highlight_widget(self.axis_style_group)
            self.plot_settings_scroll_area.ensureWidgetVisible(self.axis_style_group)
            self.axis_border_width_spinbox.setFocus()
            return

        for artist in reversed(self.ax.get_children()):
            if not isinstance(artist, (plt.Artist)) or artist in self.draggable_handlers or artist == self.ax.patch: continue
            
            contains, details = artist.contains(event)
            if contains and (artist_info := self.artists_map.get(artist)):
                self.control_tabs.setCurrentWidget(self.plot_settings_tab)
                self.highlight_widget(self.style_group)
                self.plot_settings_scroll_area.ensureWidgetVisible(self.style_group)
                
                try:
                    self.is_updating_ui = True
                    ds_idx, artist_type = artist_info['dataset_index'], artist_info['type']
                    current_ds = self.datasets[ds_idx]
                    point_idx = -1
                    
                    if artist_type == 'line': point_idx = np.argmin(np.sum((artist.get_xydata() - [event.xdata, event.ydata])**2, axis=1))
                    elif artist_type == 'scatter': point_idx = details.get("ind", [0])[0]
                    elif artist_type == 'bar': point_idx = (np.abs(np.array(current_ds['x']) - (artist.get_x() + artist.get_width() / 2))).argmin()
                    
                    if point_idx != -1:
                        self.selected_artist_info = {'type': artist_type, 'dataset_index': ds_idx, 'point_index': point_idx}
                        if (idx := self.marker_combo.findText(current_ds.get('marker', '無'))) != -1: self.marker_combo.setCurrentIndex(idx)
                        if (idx := self.linestyle_combo.findText(current_ds.get('linestyle', '實線'))) != -1: self.linestyle_combo.setCurrentIndex(idx)
                        self.line_width_spinbox.setValue(current_ds.get('linewidth', 2.0))
                        self.line_color_hex = current_ds.get('primary_color', '#1f77b4')
                        self.point_color_hex = current_ds.get('colors', [self.point_color_hex])[0]
                        self.border_color_hex = current_ds.get('border_color', '#000000')
                        self.update_button_color()
                        self.update_series_combo() 
                finally:
                    self.is_updating_ui = False
                return

        if any(annot.get_visible() and annot.contains(event)[0] for annot in self.annotations):
            return
        
        self.clear_all_highlights()

    def on_press_annotate(self, event):
        if event.button == 1 and event.inaxes:
            for annot in self.annotations:
                if annot.get_visible() and annot.contains(event)[0] and hasattr(annot, 'my_id'):
                    self.dragged_annotation = annot
                    
                    if annot.get_anncoords() == 'offset points':
                        xy_pixels = self.ax.transData.transform(annot.xy)
                        text_pos_pixels = xy_pixels + np.array(annot.xytext)
                        annot.set_position(self.ax.transData.inverted().transform(text_pos_pixels))
                        annot.set_anncoords('data')
                    
                    text_pos_pixels = self.ax.transData.transform(annot.get_position())
                    self.drag_start_offset = (text_pos_pixels[0] - event.x, text_pos_pixels[1] - event.y)
                    break

    def on_motion_annotate(self, event):
        if self.dragged_annotation and event.inaxes and event.x is not None:
            x_pixel_new = event.x + self.drag_start_offset[0]
            y_pixel_new = event.y + self.drag_start_offset[1]
            
            new_pos_data = self.ax.transData.inverted().transform((x_pixel_new, y_pixel_new))
            self.dragged_annotation.set_position(new_pos_data)
            self.canvas.draw_idle()

    def on_release_annotate(self, event):
        if self.dragged_annotation and hasattr(self.dragged_annotation, 'my_id'):
            self.annotation_positions[self.dragged_annotation.my_id] = self.dragged_annotation.get_position()
        self.dragged_annotation = None

    def update_button_color(self):
        self.line_color_btn.setStyleSheet(f"background-color: {self.line_color_hex};")
        self.point_color_btn.setStyleSheet(f"background-color: {self.point_color_hex};")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color_hex};")
        self.major_grid_color_btn.setStyleSheet(f"background-color: {self.major_grid_color_hex};")
        self.minor_grid_color_btn.setStyleSheet(f"background-color: {self.minor_grid_color_hex};")
        self.border_color_btn.setStyleSheet(f"background-color: {self.border_color_hex};")
        self.x_label_color_btn.setStyleSheet(f"background-color: {self.x_label_color_hex};")
        self.y_label_color_btn.setStyleSheet(f"background-color: {self.y_label_color_hex};")
        
    def pick_color(self, target):
        initial_color_attr = f"{target}_color_hex"
        initial_color = getattr(self, initial_color_attr, self.line_color_hex)
        color = QColorDialog.getColor(initial=QColor(initial_color))
        
        if color.isValid():
            hex_color = color.name()
            
            if target == "line":
                self.line_color_hex = hex_color
                if self.selected_artist_info and (ds_index := self.selected_artist_info['dataset_index']) < len(self.datasets):
                    ds = self.datasets[ds_index]
                    ds['primary_color'] = hex_color
                    ds['line_segment_colors'] = [hex_color] * (len(ds.get('x', [])) - 1)
                else:
                    for ds in self.datasets:
                        ds['primary_color'] = hex_color
                        ds['line_segment_colors'] = [hex_color] * (len(ds.get('x', [])) - 1)
            
            elif target == "point":
                self.point_color_hex = hex_color
                if self.selected_artist_info and (ds_index := self.selected_artist_info['dataset_index']) < len(self.datasets):
                    ds = self.datasets[ds_index]
                    ds['colors'] = [hex_color] * len(ds['colors'])
                else:
                    for ds in self.datasets:
                        ds['colors'] = [hex_color] * len(ds.get('colors', []))
                self.update_table()
            
            elif target == "border":
                self.border_color_hex = hex_color
                if self.selected_artist_info and (ds_index := self.selected_artist_info['dataset_index']) < len(self.datasets):
                    self.datasets[ds_index]['border_color'] = hex_color
                else:
                    for ds in self.datasets: ds['border_color'] = hex_color
            
            else:
                setattr(self, initial_color_attr, hex_color)

            self.update_button_color()
            self.update_plot()

    def update_dataset_colors(self, new_color, update_all=False):
        for dataset in self.datasets:
            dataset['primary_color'] = new_color
            if update_all:
                dataset['colors'] = [new_color] * len(dataset.get('colors', []))
                dataset['line_segment_colors'] = [new_color] * (len(dataset.get('x', [])) - 1)
        if update_all: self.update_table()
        self.update_plot()

    def pick_color_for_cell(self, row, col):
        if not self.is_updating_table and col > 0 and (col % 2 == 0):
            dataset_index = (col // 2) - 1
            if dataset_index >= len(self.datasets): return
            color = QColorDialog.getColor()
            if color.isValid() and row < len(self.datasets[dataset_index]['colors']):
                hex_color = color.name()
                self.datasets[dataset_index]['colors'][row] = hex_color
                self.data_table.item(row, col).setBackground(QColor(hex_color))
                self.update_plot()
                
    def update_data_from_table(self):
        if self.is_updating_table: return
        self.data_source = 'manual'
        
        try:
            x_data_str = [self.data_table.item(r, 0).text().strip() if self.data_table.item(r, 0) else "0" for r in range(self.data_table.rowCount())]
            x_data = [float(x) if x.replace('.', '', 1).lstrip('-').isdigit() else x for x in x_data_str]
        except (ValueError, AttributeError): x_data = x_data_str

        new_datasets = []
        for i in range((self.data_table.columnCount() - 1) // 2):
            y_col, c_col = 1 + i * 2, 2 + i * 2
            y_data_str = [self.data_table.item(r, y_col).text().strip() if self.data_table.item(r, y_col) else "0" for r in range(self.data_table.rowCount())]
            y_data = [float(y) if y.replace('.', '', 1).lstrip('-').isdigit() else y for y in y_data_str]
            colors = [self.data_table.item(r, c_col).background().color().name() if self.data_table.item(r, c_col) else self.point_color_hex for r in range(self.data_table.rowCount())]
            
            old_ds = self.datasets[i] if i < len(self.datasets) else {}
            new_ds = old_ds.copy()
            new_ds.update({
                'name': self.data_table.horizontalHeaderItem(y_col).text().replace("Y: ", ""),
                'x': x_data[:len(y_data)], 'y': y_data, 'colors': colors, 
                'primary_color': old_ds.get('primary_color', self.line_color_hex)
            })
            new_datasets.append(new_ds)
        
        self.datasets = new_datasets
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_plot()
        self.update_series_combo() 

    def on_table_sort(self, column, order):
        if self.last_sort_info['col'] == column: self.last_sort_info['count'] += 1
        else: self.last_sort_info = {'col': column, 'count': 1}
        if self.last_sort_info['count'] >= 3:
            self.data_table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
            self.datasets = [ds.copy() for ds in self.original_datasets]
            self.update_table(); self.update_plot(); self.last_sort_info['col'] = -1
            return
        self.update_data_from_table()

    def update_data_from_file_input(self):
        if self.excel_data is None: return
        self.data_source = 'file'
        x_col = self.x_col_combo.currentText()
        y_cols = [item.text() for item in self.y_col_list.selectedItems()]
        self.annotation_positions.clear()
        self.datasets = []
        if not x_col or not y_cols: self.update_plot(); self.update_table(); return
        try:
            x_data = self.excel_data[x_col].tolist()
            if not self.x_label_input.text(): self.x_label_input.setText(x_col)
            for y_col in y_cols:
                y_data = self.excel_data[y_col].tolist()
                n = len(y_data)
                self.datasets.append({
                    'name': y_col, 'x': x_data[:n], 'y': y_data, 
                    'colors': [self.point_color_hex] * n, 'primary_color': self.line_color_hex, 
                    'line_segment_colors': [self.line_color_hex] * (n - 1), 'marker': '圓形', 
                    'border_color': '#000000', 'linewidth': self.line_width_spinbox.value(),
                    'linestyle': self.linestyle_combo.currentText()
                })
            self.original_datasets = [ds.copy() for ds in self.datasets]
            self.y_label_input.setText(y_cols[0] if len(y_cols) == 1 else "數據值")
            self.update_plot(); self.update_table()
            self.update_series_combo()
        except Exception as e:
            QMessageBox.critical(self, "數據讀取錯誤", f"選擇的欄位有問題。\n{e}")

    def load_excel_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "選擇檔案", "", "Excel/CSV (*.xlsx *.xls *.csv)")
        if filename:
            try:
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.xlsx', '.xls']: self.excel_data = pd.read_excel(filename)
                elif ext == '.csv':
                    try: self.excel_data = pd.read_csv(filename, encoding='utf-8')
                    except UnicodeDecodeError: self.excel_data = pd.read_csv(filename, encoding='big5')
                
                cols = self.excel_data.columns.tolist()
                self.x_col_combo.clear(); self.y_col_list.clear()
                self.x_col_combo.addItems(cols); self.y_col_list.addItems(cols)
                if len(cols) >= 2: self.x_col_combo.setCurrentIndex(0); self.y_col_list.setCurrentRow(1)
                self.data_source = 'file'
            except Exception as e: QMessageBox.critical(self, "讀取錯誤", f"無法讀取檔案：{e}")

    def add_row(self):
        if not self.datasets:
            self.datasets.append({'name': '數據1', 'x': [0], 'y': [0], 'colors': [self.point_color_hex], 'primary_color': self.line_color_hex, 'line_segment_colors': [], 'marker': '圓形', 'border_color': '#000000', 'linewidth': self.line_width_spinbox.value(), 'linestyle': self.linestyle_combo.currentText()})
        else:
            for ds in self.datasets:
                last_x = ds['x'][-1] if ds['x'] and isinstance(ds['x'][-1], (int, float)) else len(ds['x']) -1
                ds['x'].append(last_x + 1); ds['y'].append(0); ds['colors'].append(self.point_color_hex)
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_table(); self.update_plot()
        self.update_series_combo()

    def remove_row(self):
        rows = sorted(list(set(index.row() for index in self.data_table.selectedIndexes())), reverse=True)
        if not rows: return
        for row in rows:
            for ds in self.datasets:
                if row < len(ds['x']):
                    for key in ['x', 'y', 'colors']: 
                        if row < len(ds[key]):
                            del ds[key][row]
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_table(); self.update_plot()
        self.update_series_combo()

    def move_row(self, direction):
        if not (sel := self.data_table.selectedIndexes()): return
        row = sel[0].row()
        new_row = row + direction
        if not (0 <= new_row < self.data_table.rowCount()): return
        
        self.is_updating_table = True
        for ds in self.datasets:
            for key in ['x', 'y', 'colors']:
                if row < len(ds[key]) and new_row < len(ds[key]):
                    ds[key][row], ds[key][new_row] = ds[key][new_row], ds[key][row]
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_table()
        self.data_table.selectRow(new_row)
        self.is_updating_table = False
        self.update_plot()

    def move_row_up(self): self.move_row(-1)
    def move_row_down(self): self.move_row(1)
        
    def clear_plot(self):
        self.datasets, self.original_datasets, self.excel_data = [], [], None
        self.data_source, self.annotation_positions = 'manual', {}
        for w in [self.title_input, self.x_label_input, self.y_label_input, self.x_col_combo, self.y_col_list]: w.clear()
        self.update_table(); self.update_plot()
        self.update_series_combo()

    def get_settings(self):
        settings = {w.objectName(): w.text() if isinstance(w, QLineEdit) else w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.isChecked() if isinstance(w, QCheckBox) else w.currentText() if isinstance(w, QComboBox) else None for w in self.findChildren((QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)) if w.objectName()}
        settings.update({k: getattr(self, k) for k in dir(self) if k.endswith("_hex")})
        settings["datasets_styles"] = [{"marker": d.get('marker'), "linewidth": d.get('linewidth'), "linestyle": d.get('linestyle')} for d in self.datasets]
        settings["x_tick_rotation"] = self.x_tick_rotation_spinbox.value()
        settings["y_tick_rotation"] = self.y_tick_rotation_spinbox.value()
        return settings

    def set_settings(self, s):
        self.is_updating_ui = True
        try:
            for k, v in s.items():
                if (widget := self.findChild(QWidget, k)):
                    if isinstance(widget, QLineEdit): widget.setText(v)
                    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(v)
                    elif isinstance(widget, QCheckBox): widget.setChecked(v)
                    elif isinstance(widget, QComboBox) and (idx := widget.findText(v)) != -1: widget.setCurrentIndex(idx)
                elif k.endswith("_hex"): setattr(self, k, v)
            if "x_tick_rotation" in s: self.x_tick_rotation_spinbox.setValue(s["x_tick_rotation"])
            if "y_tick_rotation" in s: self.y_tick_rotation_spinbox.setValue(s["y_tick_rotation"])
            if "datasets_styles" in s:
                for i, style in enumerate(s["datasets_styles"]):
                    if i < len(self.datasets): self.datasets[i].update(style)
        finally:
            self.is_updating_ui = False
        self.update_button_color()
        self.toggle_plot_settings()

    def create_collapsible_container(self, title, content_layout, obj_name=None):
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(container); layout.setContentsMargins(5, 5, 5, 5)
        btn = QPushButton(f"▼ {title}")
        btn.setStyleSheet("text-align: left; font-weight: bold; border: none; background-color: #e0e0e0;")
        content = QWidget(); content.setLayout(content_layout)
        btn.clicked.connect(lambda: (content.setVisible(not content.isVisible()), btn.setText(f"{'▼' if content.isVisible() else '▶'} {title}")))
        layout.addWidget(btn); layout.addWidget(content)
        if obj_name: container.setObjectName(obj_name)
        return container
        
    def highlight_widget(self, widget, duration=1000):
        original_style = widget.styleSheet()
        widget.setStyleSheet(original_style + " QFrame { border: 2px solid #add8e6; }")
        QTimer.singleShot(duration, lambda: widget.setStyleSheet(original_style))
    
    def clear_all_highlights(self):
        """ Clears all active highlights from widgets. """
        for widget, original_style in list(self.highlighted_widgets):
            try:
                widget.setStyleSheet(original_style)
            except RuntimeError:
                pass
        self.highlighted_widgets.clear()

    def table_key_press_event(self, event):
        if event.matches(QKeySequence.StandardKey.Copy): self.copy_data()
        elif event.matches(QKeySequence.StandardKey.Paste): self.paste_data()
        else: QTableWidget.keyPressEvent(self.data_table, event)
    
    def copy_data(self):
        if not (sel := self.data_table.selectedRanges()): return
        text = '\n'.join(['\t'.join([self.data_table.item(r, c).text() if self.data_table.item(r, c) else '' for c in range(sel[0].leftColumn(), sel[0].rightColumn() + 1)]) for r in range(sel[0].topRow(), sel[0].bottomRow() + 1)])
        QApplication.clipboard().setText(text)

    def paste_data(self):
        text, sel = QApplication.clipboard().text(), self.data_table.selectedIndexes()
        start_row = sel[0].row() if sel else 0
        start_col = sel[0].column() if sel else 0
        lines = text.strip('\n').split('\n')
        self.is_updating_table = True
        for i, line in enumerate(lines):
            row = start_row + i
            if row >= self.data_table.rowCount(): self.data_table.insertRow(row)
            for j, field in enumerate(line.split('\t')):
                if (col := start_col + j) < self.data_table.columnCount(): self.data_table.setItem(row, col, QTableWidgetItem(field.strip()))
        self.is_updating_table = False; self.update_data_from_table()
        
    def filter_table(self, text):
        for r in range(self.data_table.rowCount()):
            self.data_table.setRowHidden(r, text.lower() not in ''.join([self.data_table.item(r, c).text().lower() for c in range(self.data_table.columnCount()) if self.data_table.item(r, c)]))

    def save_template(self):
        filename, _ = QFileDialog.getSaveFileName(self, "儲存範本", "", "JSON (*.json)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f: json.dump(self.get_settings(), f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "成功", "範本已儲存。")
            except Exception as e: QMessageBox.critical(self, "錯誤", f"無法儲存檔案：{e}")

    def load_template(self):
        filename, _ = QFileDialog.getOpenFileName(self, "載入範本", "", "JSON (*.json)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f: self.set_settings(json.load(f))
                QMessageBox.information(self, "成功", "範本已載入。")
            except Exception as e: QMessageBox.critical(self, "錯誤", f"無法載入檔案：{e}")

if __name__ == '__main__':
    # <--- 新增: 為了跨平台（特別是Windows）使用 multiprocessing 的必要保護 --->
    multiprocessing.freeze_support()
    try:
        app = QApplication(sys.argv)
        main_window = PlottingApp()
        main_window.show()
        sys.exit(app.exec())
    except ImportError as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"缺少必要的函式庫: {e.name}\n\n請在終端機或命令提示字元中執行以下指令來安裝：\npip install {e.name}")
        msg.setWindowTitle("啟動失敗")
        msg.exec()
    except Exception as e:
        import traceback
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"發生未預期的錯誤: {e}")
        msg.setInformativeText(traceback.format_exc())
        msg.setWindowTitle("程式啟動錯誤")
        msg.exec()