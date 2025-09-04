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
from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import QColor, QKeySequence
import os
from zipfile import BadZipFile
import json
import numpy as np
import time

# 檢查 scipy 是否存在，並處理 ImportError
try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: 找不到 scipy 函式庫，平滑曲線功能將不可用。請使用 'pip install scipy' 進行安裝。")


# [V3.2.0 新增] 輔助類別，用於使 Matplotlib 的 artist (如文字) 可拖曳
class DraggableArtist:
    """ 一個讓 Matplotlib artist (例如 Title, Label) 可被滑鼠拖曳的類別。 """
    def __init__(self, artist, canvas):
        self.artist = artist
        self.canvas = canvas
        self.is_dragging = False
        self.press_offset = (0, 0)
        
        # 連接滑鼠事件
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        """ 處理滑鼠按下事件 """
        if event.inaxes is None or self.is_dragging:
            return
        
        # 檢查點擊是否在 artist 的邊界框內
        contains, _ = self.artist.contains(event)
        if not contains:
            return
            
        self.is_dragging = True
        # 獲取 artist 的目前位置 (以數據座標為單位)
        pos = self.artist.get_position()
        # 計算點擊位置與 artist 位置的偏移
        self.press_offset = (pos[0] - event.xdata, pos[1] - event.ydata)

    def on_motion(self, event):
        """ 處理滑鼠移動事件 """
        if not self.is_dragging or event.inaxes is None:
            return
            
        # 根據滑鼠目前位置和初始偏移，計算新的 artist 位置
        x = event.xdata + self.press_offset[0]
        y = event.ydata + self.press_offset[1]
        self.artist.set_position((x, y))
        
        # 通知畫布重新繪製
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

        self.setWindowTitle("多功能繪圖工具-V3.2.1")
        self.setGeometry(100, 100, 1200, 800)
        
        # 設置 Matplotlib 字體以支援中文顯示
        self.set_matplotlib_font()
        
        # Matplotlib 圖形設定
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.legend = None

        self.excel_data = None
        self.data_source = 'manual'
        self.is_updating_table = False
        self.is_updating_ui = False # 新增旗標，避免介面更新導致無限迴圈
        self.original_datasets = [] # 用於恢復排序
        self.last_sort_info = {'col': -1, 'count': 0}
        
        # 儲存當前選取的物件資訊
        self.selected_artist_info = None # {'type': 'line'/'scatter'..., 'dataset_index': index, 'point_index': index}
        
        # 用於多條曲線的數據結構
        self.datasets = []
        self.artists_map = {} # 儲存繪圖物件與其索引的對應關係

        # 用於拖曳數據標籤的變數
        self.annotations = []
        self.dragged_annotation = None
        self.drag_start_offset = (0, 0)
        self.annotation_positions = {}
        
        # [V3.2.0 新增] 用於存放可拖曳物件的處理器
        self.draggable_handlers = []
        
        # [V3.2.1 新增] 用於連續移動的計時器
        self.move_up_timer = QTimer(self)
        self.move_down_timer = QTimer(self)
        
        # 用於管理高亮
        self.highlighted_widgets = []

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

        # 連結滑鼠事件
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_press_event', self.on_press_annotate)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_annotate)
        self.canvas.mpl_connect('button_release_event', self.on_release_annotate)
        
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()

        # 初始化更新
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
        
        self.info_label = QLabel(" ")
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.info_label.setStyleSheet("background-color: rgba(255, 255, 255, 0.7); padding: 2px; border-radius: 3px;")
        
        toolbar_info_layout = QHBoxLayout()
        toolbar_info_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_info_layout.addWidget(self.toolbar)
        toolbar_info_layout.addStretch()
        toolbar_info_layout.addWidget(self.info_label)
        
        # [V3.2.1 修改] 設定伸縮因子，確保畫布最大化
        plot_area_layout.addLayout(toolbar_info_layout, 0)
        plot_area_layout.addWidget(self.canvas, 1)
        
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
        
        self.y_col_label = QLabel("Y 欄位 (可多選):")
        excel_layout.addWidget(self.y_col_label, 2, 0, 1, 2)
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
        data_table_layout.addWidget(self.data_table)

        table_control_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("新增行")
        self.add_row_btn.clicked.connect(self.add_row)
        self.remove_row_btn = QPushButton("刪除行")
        self.remove_row_btn.clicked.connect(self.remove_row)
        self.move_up_btn = QPushButton("上移")
        self.move_down_btn = QPushButton("下移")
        
        # [V3.2.1 修改] 連接 pressed 和 released 信號以實現連續移動
        self.move_up_btn.pressed.connect(self.start_moving_up)
        self.move_up_btn.released.connect(self.stop_moving)
        self.move_down_btn.pressed.connect(self.start_moving_down)
        self.move_down_btn.released.connect(self.stop_moving)
        self.move_up_timer.timeout.connect(self.move_row_up)
        self.move_down_timer.timeout.connect(self.move_row_down)

        table_control_layout.addWidget(self.add_row_btn)
        table_control_layout.addWidget(self.remove_row_btn)
        table_control_layout.addWidget(self.move_up_btn)
        table_control_layout.addWidget(self.move_down_btn)
        data_table_layout.addLayout(table_control_layout)
        data_table_group = self.create_collapsible_container("數據表格", data_table_layout)
        data_settings_layout.addWidget(data_table_group)

        # 繪圖設定分頁的內容
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
        self.x_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
        self.x_label_input.textChanged.connect(self.update_plot)
        self.settings_layout.addWidget(self.x_label_input)

        self.settings_layout.addWidget(QLabel("Y 軸標籤:"))
        self.y_label_input = QLineEdit()
        self.y_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
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
        self.show_data_labels_checkbox.setChecked(False)
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
        self.x_decimal_spinbox.setMinimum(0)
        self.x_decimal_spinbox.setMaximum(999999999)
        self.x_decimal_spinbox.setValue(2)
        self.x_decimal_spinbox.valueChanged.connect(self.update_plot)
        decimal_layout.addWidget(self.x_decimal_spinbox)
        self.settings_layout.addLayout(decimal_layout)
        
        decimal_layout_y = QHBoxLayout()
        decimal_layout_y.addWidget(QLabel("Y軸小數點:"))
        self.y_decimal_spinbox = QSpinBox()
        self.y_decimal_spinbox.setMinimum(0)
        self.y_decimal_spinbox.setMaximum(999999999)
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

        axis_border_width_layout = QHBoxLayout()
        axis_border_width_layout.addWidget(QLabel("座標軸邊框粗度:"))
        self.axis_border_width_spinbox = QDoubleSpinBox()
        self.axis_border_width_spinbox.setMinimum(0.0)
        self.axis_border_width_spinbox.setMaximum(float('inf'))
        self.axis_border_width_spinbox.setValue(1.0)
        self.axis_border_width_spinbox.setSingleStep(0.5)
        self.axis_border_width_spinbox.valueChanged.connect(self.update_plot)
        axis_border_width_layout.addWidget(self.axis_border_width_spinbox)
        self.axis_border_width_spinbox.valueChanged.connect(self.update_plot)
        axis_style_layout.addLayout(axis_border_width_layout)

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
        self.major_tick_length_spinbox.setMinimum(0.0)
        self.major_tick_length_spinbox.setValue(3.5)
        self.major_tick_length_spinbox.valueChanged.connect(self.update_plot)
        major_tick_size_layout.addWidget(self.major_tick_length_spinbox, 0, 1)

        major_tick_size_layout.addWidget(QLabel("寬度:"), 0, 2)
        self.major_tick_width_spinbox = QDoubleSpinBox()
        self.major_tick_width_spinbox.setMinimum(0.0)
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
        self.minor_grid_checkbox.setChecked(False)
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
        self.minor_x_interval_spinbox.setMinimum(0.0)
        self.minor_x_interval_spinbox.setMaximum(float('inf'))
        self.minor_x_interval_spinbox.setValue(0.5)
        self.minor_x_interval_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_layout.addWidget(self.minor_x_interval_spinbox)
        
        minor_tick_layout.addWidget(QLabel("次刻度線 Y 軸間隔:"))
        self.minor_y_interval_spinbox = QDoubleSpinBox()
        self.minor_y_interval_spinbox.setMinimum(0.0)
        self.minor_y_interval_spinbox.setMaximum(float('inf'))
        self.minor_y_interval_spinbox.setValue(0.5)
        self.minor_y_interval_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_layout.addWidget(self.minor_y_interval_spinbox)

        minor_tick_size_layout = QGridLayout()
        minor_tick_size_layout.addWidget(QLabel("次刻度線 長度:"), 0, 0)
        self.minor_tick_length_spinbox = QDoubleSpinBox()
        self.minor_tick_length_spinbox.setMinimum(0.0)
        self.minor_tick_length_spinbox.setValue(2.0)
        self.minor_tick_length_spinbox.valueChanged.connect(self.update_plot)
        minor_tick_size_layout.addWidget(self.minor_tick_length_spinbox, 0, 1)
        
        minor_tick_size_layout.addWidget(QLabel("寬度:"), 0, 2)
        self.minor_tick_width_spinbox = QDoubleSpinBox()
        self.minor_tick_width_spinbox.setMinimum(0.0)
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
        
    def setup_dynamic_widgets(self):
        """ 為動態顯示的組件建立並添加布局。 """
        self.style_layout.addWidget(self.connect_scatter_checkbox)
        self.line_width_label = QLabel("線條寬度:")
        self.style_layout.addWidget(self.line_width_label)
        self.line_width_spinbox.setMinimum(0.5)
        self.line_width_spinbox.setMaximum(float('inf'))
        self.line_width_spinbox.setValue(2.0)
        self.line_width_spinbox.valueChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.line_width_spinbox)
        self.bar_width_label = QLabel("長條圖寬度:")
        self.style_layout.addWidget(self.bar_width_label)
        self.bar_width_spinbox.setMinimum(0.01)
        self.bar_width_spinbox.setMaximum(float('inf'))
        self.bar_width_spinbox.setSingleStep(0.05)
        self.bar_width_spinbox.setValue(0.8)
        self.bar_width_spinbox.valueChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.bar_width_spinbox)
        self.point_size_label = QLabel("點的大小:")
        self.style_layout.addWidget(self.point_size_label)
        self.point_size_spinbox.setMinimum(1.0)
        self.point_size_spinbox.setMaximum(float('inf'))
        self.point_size_spinbox.setValue(10.0)
        self.point_size_spinbox.valueChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.point_size_spinbox)
        self.linestyle_label = QLabel("線條樣式:")
        self.style_layout.addWidget(self.linestyle_label)
        self.linestyle_combo.addItems(["實線", "虛線", "點虛線", "點"])
        self.linestyle_combo.currentIndexChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.linestyle_combo)
        self.marker_label = QLabel("標記樣式:")
        self.style_layout.addWidget(self.marker_label)
        self.marker_combo.addItems(["圓形", "方形", "三角形", "星形", "無"])
        self.marker_combo.currentIndexChanged.connect(self.update_plot)
        self.style_layout.addWidget(self.marker_combo)
        if SCIPY_AVAILABLE:
            self.style_layout.addWidget(self.smooth_line_checkbox)

    def update_plot(self):
        """ 根據當前數據和設定更新繪圖。 """
        if self.is_updating_ui:
            return
            
        # [V3.2.0 修改] 在重繪前，清理舊的拖曳事件處理器
        for handler in self.draggable_handlers:
            handler.disconnect()
        self.draggable_handlers.clear()
        
        self.ax.clear()
        self.annotations.clear()
        self.artists_map.clear()
        self.figure.set_facecolor(self.bg_color_hex)
        self.ax.set_facecolor(self.bg_color_hex)

        if not self.datasets or (self.data_source == 'file' and not self.y_col_list.selectedItems()):
            self.ax.set_title("請輸入或選擇數據以繪製圖表")
            self.canvas.draw()
            return
            
        for dataset_index, dataset in enumerate(self.datasets):
            x_data, y_data, colors, name = dataset['x'], dataset['y'], dataset['colors'], dataset['name']
            primary_color = dataset.get('primary_color', self.plot_color_hex)
            line_segment_colors = dataset.get('line_segment_colors', [primary_color] * (len(x_data) - 1))
            y_is_numeric = all(isinstance(y, (int, float)) for y in y_data)
            if not y_data: continue

            if self.line_checkbox.isChecked():
                linestyle_map = {"實線": "-", "虛線": "--", "點虛線": "-.", "點": ":"}
                ls = linestyle_map.get(self.linestyle_combo.currentText(), '-')
                for i in range(len(x_data) - 1):
                    line, = self.ax.plot(x_data[i:i+2], y_data[i:i+2], color=line_segment_colors[i], 
                                         linewidth=self.line_width_spinbox.value(), linestyle=ls, zorder=1)
                    self.artists_map[line] = (dataset_index, i)
                self.ax.plot([], [], color=primary_color, linewidth=self.line_width_spinbox.value(), linestyle=ls, label=name)

            if self.scatter_checkbox.isChecked() or (self.line_checkbox.isChecked() and self.marker_combo.currentText() != "無"):
                marker_map = {"圓形": "o", "方形": "s", "三角形": "^", "星形": "*", "無": "None"}
                marker = marker_map.get(self.marker_combo.currentText(), 'o')
                if marker != "None":
                    scatter = self.ax.scatter(x_data, y_data, s=self.point_size_spinbox.value()**2, marker=marker,
                                              c=colors, edgecolors=self.border_color_hex, 
                                              linewidths=self.border_width_spinbox.value(), zorder=2)
                    self.artists_map[scatter] = dataset_index
            
            if self.bar_checkbox.isChecked():
                bars = self.ax.bar(x_data, y_data, width=self.bar_width_spinbox.value(), color=colors,
                                   edgecolor=self.border_color_hex, linewidth=self.border_width_spinbox.value(), 
                                   zorder=2, label=name)
                for rect in bars: self.artists_map[rect] = dataset_index
                    
            if self.box_checkbox.isChecked():
                box_plot = self.ax.boxplot(y_data, patch_artist=True)
                for patch in box_plot['boxes']: patch.set_facecolor(primary_color)
                self.ax.set_xticks([dataset_index + 1])
                self.ax.set_xticklabels([name])
                
            if self.show_data_labels_checkbox.isChecked():
                if not self.box_checkbox.isChecked():
                    for i, (x, y, color) in enumerate(zip(x_data, y_data, colors)):
                        label_parts = []
                        if self.show_x_labels_checkbox.isChecked(): label_parts.append(f"{x:.{self.x_decimal_spinbox.value()}f}" if isinstance(x, (int, float)) else f"{x}")
                        if self.show_y_labels_checkbox.isChecked(): label_parts.append(f"{y:.{self.y_decimal_spinbox.value()}f}" if isinstance(y, (int, float)) else f"{y}")
                        label_text = ", ".join(label_parts)
                        if label_text:
                            my_id = (dataset_index, i)
                            if my_id in self.annotation_positions:
                                pos = self.annotation_positions[my_id]
                                annot = self.ax.annotate(label_text, (x, y), xytext=pos, textcoords='data', ha='center',
                                                         fontsize=self.data_label_size_spinbox.value(), color=color)
                            else:
                                annot = self.ax.annotate(label_text, (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                                                         fontsize=self.data_label_size_spinbox.value(), color=color)
                            annot.my_id = my_id
                            self.annotations.append(annot)
                elif y_is_numeric:
                    median_val = np.median(y_data)
                    annot = self.ax.annotate(f"中位數: {median_val:.{self.y_decimal_spinbox.value()}f}", 
                                             xy=(1, median_val), xytext=(10, 0), textcoords='offset points',
                                             fontsize=self.data_label_size_spinbox.value(), color='black')
                    self.annotations.append(annot)

        # 全局圖表設定
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

        plt.setp(self.ax.get_xticklabels(), fontweight='bold' if self.x_tick_label_bold_checkbox.isChecked() else 'normal')
        plt.setp(self.ax.get_yticklabels(), fontweight='bold' if self.y_tick_label_bold_checkbox.isChecked() else 'normal')

        if self.datasets and self.datasets[0]['x'] and all(isinstance(x, (int, float)) for x in self.datasets[0]['x']):
            if self.x_interval_spinbox.value() > 0 and not self.box_checkbox.isChecked(): self.ax.xaxis.set_major_locator(ticker.MultipleLocator(self.x_interval_spinbox.value()))
            if self.minor_x_interval_spinbox.value() > 0 and not self.box_checkbox.isChecked(): self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(self.minor_x_interval_spinbox.value()))
        
        if self.datasets and self.datasets[0]['y'] and all(isinstance(y, (int, float)) for y in self.datasets[0]['y']):
            if self.y_interval_spinbox.value() > 0: self.ax.yaxis.set_major_locator(ticker.MultipleLocator(self.y_interval_spinbox.value()))
            if self.minor_y_interval_spinbox.value() > 0: self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(self.minor_y_interval_spinbox.value()))

        for spine in self.ax.spines.values():
            spine.set_linewidth(self.axis_border_width_spinbox.value())
        
        self.ax.grid(False, which='both')
        if self.major_grid_checkbox.isChecked():
            self.ax.grid(True, which='major', color=self.major_grid_color_hex, linestyle='-', linewidth=0.5)
        
        if self.minor_grid_checkbox.isChecked():
            self.ax.minorticks_on()
            self.ax.grid(True, which='minor', color=self.minor_grid_color_hex, linestyle=':', linewidth=0.5)
            self.ax.tick_params(axis='both', which='minor',
                                length=self.minor_tick_length_spinbox.value(),
                                width=self.minor_tick_width_spinbox.value())
        else:
            self.ax.minorticks_off()

        if self.datasets and any([self.line_checkbox.isChecked(), self.bar_checkbox.isChecked(), self.box_checkbox.isChecked(), self.scatter_checkbox.isChecked()]):
            self.legend = self.ax.legend(prop={'size': self.legend_size_spinbox.value()}, draggable=True)
        else:
            self.legend = None

        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_table(self):
        """
        根據當前數據更新表格。
        """
        self.is_updating_table = True
        
        if not self.datasets:
            self.data_table.clear()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(3)
            self.data_table.setHorizontalHeaderLabels(["X 數據", "Y 數據", "顏色"])
            self.is_updating_table = False
            return

        num_datasets = len(self.datasets)
        column_count = 1 + num_datasets * 2
        self.data_table.setColumnCount(column_count)
        
        headers = ["X 數據"]
        for dataset in self.datasets:
            headers.extend([f"Y: {dataset['name']}", "顏色"])
        self.data_table.setHorizontalHeaderLabels(headers)
        
        row_count = len(self.datasets[0]['x']) if self.datasets and self.datasets[0]['x'] else 0
        self.data_table.setRowCount(row_count)
        
        for row in range(row_count):
            x_val = str(self.datasets[0]['x'][row])
            self.data_table.setItem(row, 0, QTableWidgetItem(x_val))
            
            for dataset_index, dataset in enumerate(self.datasets):
                y_col = 1 + dataset_index * 2
                color_col = y_col + 1
                
                y_val = str(dataset['y'][row])
                color_val = dataset['colors'][row]
                
                y_item = QTableWidgetItem(y_val)
                color_item = QTableWidgetItem('')
                color_item.setBackground(QColor(color_val))
                color_item.setFlags(color_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.data_table.setItem(row, y_col, y_item)
                self.data_table.setItem(row, color_col, color_item)

        self.is_updating_table = False
            
    def toggle_plot_settings(self):
        is_line_checked = self.line_checkbox.isChecked()
        is_scatter_checked = self.scatter_checkbox.isChecked()
        is_bar_checked = self.bar_checkbox.isChecked()
        self.connect_scatter_checkbox.setVisible(is_scatter_checked and not is_line_checked)
        self.line_width_spinbox.setVisible(is_line_checked or (is_scatter_checked and self.connect_scatter_checkbox.isChecked()))
        self.linestyle_combo.setVisible(is_line_checked or (is_scatter_checked and self.connect_scatter_checkbox.isChecked()))
        self.smooth_line_checkbox.setVisible(is_line_checked and SCIPY_AVAILABLE)
        self.point_size_spinbox.setVisible(is_scatter_checked or is_line_checked)
        self.marker_combo.setVisible(is_scatter_checked or is_line_checked)
        self.bar_width_spinbox.setVisible(is_bar_checked)
        self.line_width_label.setVisible(is_line_checked or (is_scatter_checked and self.connect_scatter_checkbox.isChecked()))
        self.bar_width_label.setVisible(is_bar_checked)
        self.point_size_label.setVisible(is_scatter_checked or is_line_checked)
        self.linestyle_label.setVisible(is_line_checked or (is_scatter_checked and self.connect_scatter_checkbox.isChecked()))
        self.marker_label.setVisible(is_scatter_checked or is_line_checked)
        self.update_plot()

    def on_canvas_click(self, event):
        self.selected_artist_info = None
        self.clear_all_highlights()
        if event.button != 1 or not event.inaxes: return
        for annot in self.annotations:
            if annot.get_visible() and annot.contains(event)[0]: return
        for artist in reversed(self.ax.get_children()):
            if not isinstance(artist, (plt.matplotlib.lines.Line2D, plt.matplotlib.collections.PathCollection, plt.matplotlib.patches.Rectangle)): continue
            if artist == self.ax.patch: continue
            if isinstance(artist, plt.matplotlib.patches.Rectangle) and artist not in self.artists_map: continue
            contains, details = artist.contains(event)
            if contains:
                self._handle_artist_click(artist, details, event)
                return
        legend = self.ax.get_legend()
        if legend and legend.get_visible() and legend.contains(event):
            self._handle_artist_click(legend, {}, event)
            return
        for spine in self.ax.spines.values():
            if spine.contains(event)[0]:
                self._handle_artist_click(spine, {}, event)
                return
    
    def _handle_artist_click(self, artist, details, event):
        self.control_tabs.setCurrentWidget(self.plot_settings_tab)
        try:
            self.is_updating_ui = True
            if isinstance(artist, (plt.matplotlib.lines.Line2D, plt.matplotlib.collections.PathCollection, plt.matplotlib.patches.Rectangle)) and artist in self.artists_map:
                artist_info = self.artists_map[artist]
                dataset_index = artist_info[0] if isinstance(artist_info, tuple) else artist_info
                point_index = -1
                artist_type = None
                if isinstance(artist, (plt.matplotlib.lines.Line2D)):
                    artist_type = 'line'
                    x, y = event.xdata, event.ydata
                    x_data, y_data = np.asarray(self.datasets[dataset_index]['x']), np.asarray(self.datasets[dataset_index]['y'])
                    distances = np.sqrt((x_data - x)**2 + (y_data - y)**2)
                    point_index = np.argmin(distances)
                elif isinstance(artist, plt.matplotlib.collections.PathCollection):
                    artist_type = 'scatter'
                    point_index = details.get("ind", [0])[0]
                elif isinstance(artist, plt.matplotlib.patches.Rectangle):
                    artist_type = 'bar'
                    x_pos = artist.get_x() + artist.get_width() / 2
                    x_data = np.array(self.datasets[dataset_index]['x'])
                    point_index = (np.abs(x_data - x_pos)).argmin()
                if point_index != -1:
                    self.selected_artist_info = {'type': artist_type, 'dataset_index': dataset_index, 'point_index': point_index}
                    self.data_table.clearSelection()
                    self.data_table.selectRow(point_index)
                self.highlight_widget(self.style_group)
                self.plot_settings_scroll_area.ensureWidgetVisible(self.style_group)
                if artist_type == 'line': self.line_width_spinbox.setFocus()
                elif artist_type == 'scatter': self.point_size_spinbox.setFocus()
                elif artist_type == 'bar': self.bar_width_spinbox.setFocus()
            elif isinstance(artist, plt.matplotlib.spines.Spine):
                self.highlight_widget(self.axis_style_group)
                self.plot_settings_scroll_area.ensureWidgetVisible(self.axis_style_group)
                self.axis_border_width_spinbox.setFocus()
            elif isinstance(artist, plt.matplotlib.legend.Legend):
                self.highlight_widget(self.settings_group)
                self.plot_settings_scroll_area.ensureWidgetVisible(self.settings_group)
                self.legend_size_spinbox.setFocus()
        finally:
            self.is_updating_ui = False
            
    def on_press_annotate(self, event):
        if event.button == 1 and event.inaxes:
            for annot in self.annotations:
                if annot.get_visible() and annot.contains(event)[0]:
                    self.dragged_annotation = annot
                    x_annot, y_annot = annot.get_position()
                    self.drag_start_offset = (x_annot - event.xdata, y_annot - event.ydata)
                    break

    def on_motion_annotate(self, event):
        if self.dragged_annotation and event.inaxes:
            x_new = event.xdata + self.drag_start_offset[0]
            y_new = event.ydata + self.drag_start_offset[1]
            self.dragged_annotation.set_position((x_new, y_new))
            self.canvas.draw_idle()

    def on_release_annotate(self, event):
        if self.dragged_annotation:
            if hasattr(self.dragged_annotation, 'my_id'):
                new_pos = self.dragged_annotation.get_position()
                self.annotation_positions[self.dragged_annotation.my_id] = new_pos
        self.dragged_annotation = None

    def update_button_color(self):
        self.plot_color_btn.setStyleSheet(f"background-color: {self.plot_color_hex};")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color_hex};")
        self.major_grid_color_btn.setStyleSheet(f"background-color: {self.major_grid_color_hex};")
        self.minor_grid_color_btn.setStyleSheet(f"background-color: {self.minor_grid_color_hex};")
        self.border_color_btn.setStyleSheet(f"background-color: {self.border_color_hex};")
        self.x_label_color_btn.setStyleSheet(f"background-color: {self.x_label_color_hex};")
        self.y_label_color_btn.setStyleSheet(f"background-color: {self.y_label_color_hex};")
        
    def pick_color(self, target):
        current_color = self.plot_color_hex
        if self.selected_artist_info:
            ds_index = self.selected_artist_info['dataset_index']
            pt_index = self.selected_artist_info['point_index']
            if ds_index < len(self.datasets): current_color = self.datasets[ds_index]['colors'][pt_index]
        color = QColorDialog.getColor(initial=QColor(current_color))
        if color.isValid():
            hex_color = color.name()
            if target == "plot":
                if self.selected_artist_info:
                    ds_index, pt_index, artist_type = self.selected_artist_info['dataset_index'], self.selected_artist_info['point_index'], self.selected_artist_info['type']
                    if ds_index < len(self.datasets):
                        if artist_type == 'line':
                            if pt_index > 0 and pt_index <= len(self.datasets[ds_index]['line_segment_colors']): self.datasets[ds_index]['line_segment_colors'][pt_index - 1] = hex_color
                        elif artist_type in ['scatter', 'bar'] and pt_index < len(self.datasets[ds_index]['colors']): self.datasets[ds_index]['colors'][pt_index] = hex_color
                        self.update_table(); self.update_plot()
                else: self.plot_color_hex = hex_color; self.update_dataset_colors(hex_color, update_all=True)
            elif target == "background": self.bg_color_hex = hex_color
            elif target == "major_grid": self.major_grid_color_hex = hex_color
            elif target == "minor_grid": self.minor_grid_color_hex = hex_color
            elif target == "border": self.border_color_hex = hex_color
            elif target == "x_label": self.x_label_color_hex = hex_color
            elif target == "y_label": self.y_label_color_hex = hex_color
            self.update_button_color()
            if target != "plot": self.update_plot()

    def update_dataset_colors(self, new_color, update_all=False):
        for dataset in self.datasets:
            dataset['primary_color'] = new_color
            if update_all:
                dataset['colors'] = [new_color] * len(dataset.get('colors', []))
                dataset['line_segment_colors'] = [new_color] * (len(dataset.get('x', [])) - 1)
        self.update_table(); self.update_plot()

    def pick_color_for_cell(self, row, col):
        if not self.is_updating_table and col > 0 and (col % 2 == 0):
            dataset_index = (col // 2) - 1
            if dataset_index >= len(self.datasets): return
            color = QColorDialog.getColor()
            if color.isValid():
                hex_color = color.name()
                if row < len(self.datasets[dataset_index]['colors']): self.datasets[dataset_index]['colors'][row] = hex_color
                item = self.data_table.item(row, col)
                if item is None: item = QTableWidgetItem(); self.data_table.setItem(row, col, item)
                item.setBackground(QColor(hex_color)); self.update_plot()
                
    def update_data_from_table(self):
        if self.is_updating_table: return
        self.data_source = 'manual'
        x_data = []
        for row in range(self.data_table.rowCount()):
            x_item = self.data_table.item(row, 0)
            try: x_data.append(float(x_item.text() if x_item else "0"))
            except (ValueError, TypeError): x_data.append(x_item.text() if x_item else "0")
        y_data_sets, color_sets, primary_colors = [], [], []
        for col_idx in range(1, self.data_table.columnCount(), 2):
            y_data, colors = [], []
            for row in range(self.data_table.rowCount()):
                y_item = self.data_table.item(row, col_idx)
                try: y_data.append(float(y_item.text() if y_item else "0"))
                except (ValueError, TypeError): y_data.append(y_item.text() if y_item else "0")
                color_item = self.data_table.item(row, col_idx + 1)
                colors.append(color_item.background().color().name() if color_item and color_item.background().color().isValid() else self.plot_color_hex)
            y_data_sets.append(y_data); color_sets.append(colors); primary_colors.append(colors[0] if colors else self.plot_color_hex)
        new_datasets = []
        for i, y_data in enumerate(y_data_sets):
            line_colors = self.datasets[i].get('line_segment_colors', [primary_colors[i]] * (len(x_data) - 1)) if i < len(self.datasets) else [primary_colors[i]] * (len(x_data) - 1)
            new_datasets.append({'name': self.data_table.horizontalHeaderItem(i*2+1).text().replace("Y: ", ""), 'x': x_data.copy(), 'y': y_data, 'colors': color_sets[i], 'primary_color': primary_colors[i], 'line_segment_colors': line_colors})
        self.datasets = new_datasets
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_plot()

    def on_table_sort(self, column, order):
        if self.last_sort_info['col'] == column: self.last_sort_info['count'] += 1
        else: self.last_sort_info = {'col': column, 'count': 1}
        if self.last_sort_info['count'] >= 3:
            self.data_table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
            self.datasets = [ds.copy() for ds in self.original_datasets]
            self.update_table(); self.update_plot(); self.last_sort_info = {'col': -1, 'count': 0}; return
        self.is_updating_table = True
        all_data = [[(item.background().color().name() if col > 0 and col % 2 == 0 and item.background().color().isValid() else (self.plot_color_hex if col > 0 and col % 2 == 0 else item.text())) if (item := self.data_table.item(row, col)) else "" for col in range(self.data_table.columnCount())] for row in range(self.data_table.rowCount())]
        self.is_updating_table = False
        if not all_data: self.update_plot(); return
        try: x_data = [float(x) if x else 0.0 for x in [d[0] for d in all_data]]
        except (ValueError, TypeError): x_data = [d[0] for d in all_data]
        num_datasets = (len(all_data[0]) - 1) // 2
        for i in range(num_datasets):
            y_col_idx = 1 + i * 2
            try: y_data = [float(y) if y else 0.0 for y in [d[y_col_idx] for d in all_data]]
            except (ValueError, TypeError): y_data = [d[y_col_idx] for d in all_data]
            if i < len(self.datasets): self.datasets[i]['x'] = x_data.copy(); self.datasets[i]['y'] = y_data; self.datasets[i]['colors'] = [d[y_col_idx + 1] for d in all_data]
        self.update_plot()

    def update_data_from_file_input(self):
        if self.excel_data is None: return
        self.data_source = 'file'; x_col_name = self.x_col_combo.currentText(); selected_y_cols = [item.text() for item in self.y_col_list.selectedItems()]
        self.annotation_positions.clear()
        self.datasets = []
        if not x_col_name or not selected_y_cols: self.update_plot(); self.update_table(); return
        try:
            x_data = self.excel_data[x_col_name].tolist()
            if not self.x_label_input.text(): self.x_label_input.setText(x_col_name)
            for y_col_name in selected_y_cols:
                y_data = self.excel_data[y_col_name].tolist()
                num_points = len(x_data)
                self.datasets.append({'name': y_col_name, 'x': x_data, 'y': y_data, 'colors': [self.plot_color_hex] * num_points, 'primary_color': self.plot_color_hex, 'line_segment_colors': [self.plot_color_hex] * (num_points - 1)})
            self.original_datasets = [ds.copy() for ds in self.datasets]
            if len(selected_y_cols) == 1 and not self.y_label_input.text(): self.y_label_input.setText(selected_y_cols[0])
            elif len(selected_y_cols) > 1 and not self.y_label_input.text(): self.y_label_input.setText("數據值")
            self.update_plot(); self.update_table()
        except Exception as e:
            self.datasets = []; self.ax.clear(); QMessageBox.critical(self, "數據讀取錯誤", f"選擇的欄位有問題。\n\n詳細錯誤：{e}"); self.ax.set_title("選擇的欄位有問題", color="red"); self.canvas.draw(); self.update_table()

    def load_excel_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "選擇 Excel 或 CSV 檔案", "", "支援的檔案 (*.xlsx *.xls *.csv)")
        if filename:
            try:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in (".xlsx", ".xls"): self.excel_data = pd.read_excel(filename, engine="openpyxl" if file_ext == ".xlsx" else "xlrd")
                elif file_ext == ".csv":
                    try: self.excel_data = pd.read_csv(filename, encoding='utf-8')
                    except UnicodeDecodeError: self.excel_data = pd.read_csv(filename, encoding='big5')
                else: QMessageBox.warning(self, "錯誤", "不支援的檔案類型。"); return
                self.x_col_combo.clear(); self.y_col_list.clear()
                all_cols = self.excel_data.columns.tolist()
                self.x_col_combo.addItems(all_cols); self.y_col_list.addItems(all_cols)
                QMessageBox.information(self, "成功", "檔案已載入成功。"); self.data_source = 'file'
                if len(all_cols) >= 2: self.x_col_combo.setCurrentIndex(0); self.y_col_list.item(1).setSelected(True)
            except BadZipFile: QMessageBox.critical(self, "讀取錯誤", "檔案已損壞或非標準格式。")
            except Exception as e: QMessageBox.critical(self, "讀取錯誤", f"無法讀取檔案：{e}")

    def add_row(self):
        if not self.datasets:
            self.datasets.append({'name': '數據1', 'x': [0], 'y': [0], 'colors': [self.plot_color_hex], 'primary_color': self.plot_color_hex, 'line_segment_colors': []})
        else:
            last_x = self.datasets[0]['x'][-1] if self.datasets[0]['x'] else 0
            new_x = last_x + 1 if isinstance(last_x, (int, float)) else 0
            for dataset in self.datasets:
                dataset['x'].append(new_x); dataset['y'].append(0); dataset['colors'].append(dataset['primary_color'])
                if len(dataset['x']) > 1: dataset['line_segment_colors'].append(dataset['primary_color'])
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_table(); self.update_plot()

    def remove_row(self):
        selected_rows = sorted(list(set(index.row() for index in self.data_table.selectedIndexes())), reverse=True)
        if not selected_rows: QMessageBox.warning(self, "警告", "請選擇要刪除的行。"); return
        for row in selected_rows:
            if self.datasets and 0 <= row < len(self.datasets[0]['x']):
                for dataset in self.datasets:
                    del dataset['x'][row]; del dataset['y'][row]; del dataset['colors'][row]
                    if row < len(dataset['line_segment_colors']): del dataset['line_segment_colors'][row]
        self.original_datasets = [ds.copy() for ds in self.datasets]
        self.update_table(); self.update_plot()

    def move_row_up(self):
        selected_rows = [index.row() for index in self.data_table.selectedIndexes()]
        if len(selected_rows) != 1 or selected_rows[0] == 0 or not self.datasets: return
        row_index = selected_rows[0]
        for dataset in self.datasets:
            dataset['x'].insert(row_index - 1, dataset['x'].pop(row_index))
            dataset['y'].insert(row_index - 1, dataset['y'].pop(row_index))
            dataset['colors'].insert(row_index - 1, dataset['colors'].pop(row_index))
        self.original_datasets = [ds.copy() for ds in self.datasets]; self.update_table(); self.update_plot(); self.data_table.selectRow(row_index - 1)

    def move_row_down(self):
        selected_rows = [index.row() for index in self.data_table.selectedIndexes()]
        if len(selected_rows) != 1 or selected_rows[0] == self.data_table.rowCount() - 1 or not self.datasets: return
        row_index = selected_rows[0]
        for dataset in self.datasets:
            dataset['x'].insert(row_index + 1, dataset['x'].pop(row_index))
            dataset['y'].insert(row_index + 1, dataset['y'].pop(row_index))
            dataset['colors'].insert(row_index + 1, dataset['colors'].pop(row_index))
        self.original_datasets = [ds.copy() for ds in self.datasets]; self.update_table(); self.update_plot(); self.data_table.selectRow(row_index + 1)
        
    def clear_plot(self):
        self.datasets = []; self.original_datasets = []; self.excel_data = None; self.data_source = 'manual'; self.annotation_positions.clear()
        self.title_input.clear(); self.x_label_input.clear(); self.y_label_input.clear()
        self.x_col_combo.clear(); self.y_col_list.clear()
        self.update_table(); self.update_plot()

    def get_settings(self):
        return {
            "title": self.title_input.text(), "x_label": self.x_label_input.text(), "y_label": self.y_label_input.text(),
            "plot_color_hex": self.plot_color_hex, "bg_color_hex": self.bg_color_hex, "major_grid_color_hex": self.major_grid_color_hex,
            "minor_grid_color_hex": self.minor_grid_color_hex, "border_color_hex": self.border_color_hex, "x_label_color_hex": self.x_label_color_hex,
            "y_label_color_hex": self.y_label_color_hex, "x_interval": self.x_interval_spinbox.value(), "y_interval": self.y_interval_spinbox.value(),
            "x_decimal": self.x_decimal_spinbox.value(), "y_decimal": self.y_decimal_spinbox.value(), "show_data_labels": self.show_data_labels_checkbox.isChecked(),
            "show_x_labels": self.show_x_labels_checkbox.isChecked(), "show_y_labels": self.show_y_labels_checkbox.isChecked(), "data_label_size": self.data_label_size_spinbox.value(),
            "is_line_checked": self.line_checkbox.isChecked(), "is_scatter_checked": self.scatter_checkbox.isChecked(), "is_bar_checked": self.bar_checkbox.isChecked(),
            "is_box_checked": self.box_checkbox.isChecked(), "line_width": self.line_width_spinbox.value(), "bar_width": self.bar_width_spinbox.value(),
            "border_width": self.border_width_spinbox.value(), "point_size": self.point_size_spinbox.value(), "x_label_size": self.x_label_size_spinbox.value(),
            "y_label_size": self.y_label_size_spinbox.value(), "x_label_bold": self.x_label_bold_checkbox.isChecked(), "y_label_bold": self.y_label_bold_checkbox.isChecked(),
            "x_tick_label_bold": self.x_tick_label_bold_checkbox.isChecked(), "y_tick_label_bold": self.y_tick_label_bold_checkbox.isChecked(), "linestyle": self.linestyle_combo.currentText(),
            "marker": self.marker_combo.currentText(), "show_major_grid": self.major_grid_checkbox.isChecked(), "show_minor_grid": self.minor_grid_checkbox.isChecked(),
            "axis_border_width": self.axis_border_width_spinbox.value(), "legend_size": self.legend_size_spinbox.value(), "tick_direction": self.tick_direction_combo.currentText(),
            "minor_x_interval": self.minor_x_interval_spinbox.value(), "minor_y_interval": self.minor_y_interval_spinbox.value(), "minor_tick_length": self.minor_tick_length_spinbox.value(),
            "smooth_line": self.smooth_line_checkbox.isChecked(), "major_tick_length": self.major_tick_length_spinbox.value(),
            "major_tick_width": self.major_tick_width_spinbox.value(), "minor_tick_width": self.minor_tick_width_spinbox.value()}

    def set_settings(self, s):
        self.is_updating_ui = True
        try:
            self.title_input.setText(s.get("title", "")); self.x_label_input.setText(s.get("x_label", "")); self.y_label_input.setText(s.get("y_label", ""))
            self.plot_color_hex = s.get("plot_color_hex", "#1f77b4"); self.bg_color_hex = s.get("bg_color_hex", "#ffffff"); self.major_grid_color_hex = s.get("major_grid_color_hex", "#cccccc")
            self.minor_grid_color_hex = s.get("minor_grid_color_hex", "#eeeeee"); self.border_color_hex = s.get("border_color_hex", "#000000"); self.x_label_color_hex = s.get("x_label_color_hex", "#000000")
            self.y_label_color_hex = s.get("y_label_color_hex", "#000000"); self.x_interval_spinbox.setValue(s.get("x_interval", 1.0)); self.y_interval_spinbox.setValue(s.get("y_interval", 1.0))
            self.x_decimal_spinbox.setValue(s.get("x_decimal", 2)); self.y_decimal_spinbox.setValue(s.get("y_decimal", 2)); self.show_data_labels_checkbox.setChecked(s.get("show_data_labels", False))
            self.show_x_labels_checkbox.setChecked(s.get("show_x_labels", True)); self.show_y_labels_checkbox.setChecked(s.get("show_y_labels", True)); self.data_label_size_spinbox.setValue(s.get("data_label_size", 10))
            self.line_checkbox.setChecked(s.get("is_line_checked", True)); self.scatter_checkbox.setChecked(s.get("is_scatter_checked", False)); self.bar_checkbox.setChecked(s.get("is_bar_checked", False))
            self.box_checkbox.setChecked(s.get("is_box_checked", False)); self.line_width_spinbox.setValue(s.get("line_width", 2.0)); self.bar_width_spinbox.setValue(s.get("bar_width", 0.8))
            self.border_width_spinbox.setValue(s.get("border_width", 1.0)); self.point_size_spinbox.setValue(s.get("point_size", 10.0)); self.x_label_size_spinbox.setValue(s.get("x_label_size", 12))
            self.y_label_size_spinbox.setValue(s.get("y_label_size", 12)); self.x_label_bold_checkbox.setChecked(s.get("x_label_bold", False)); self.y_label_bold_checkbox.setChecked(s.get("y_label_bold", False))
            self.x_tick_label_bold_checkbox.setChecked(s.get("x_tick_label_bold", False)); self.y_tick_label_bold_checkbox.setChecked(s.get("y_tick_label_bold", False)); self.legend_size_spinbox.setValue(s.get("legend_size", 10))
            if (idx := self.linestyle_combo.findText(s.get("linestyle", "實線"))) != -1: self.linestyle_combo.setCurrentIndex(idx)
            if (idx := self.marker_combo.findText(s.get("marker", "圓形"))) != -1: self.marker_combo.setCurrentIndex(idx)
            self.major_grid_checkbox.setChecked(s.get("show_major_grid", True)); self.minor_grid_checkbox.setChecked(s.get("show_minor_grid", False)); self.axis_border_width_spinbox.setValue(s.get("axis_border_width", 1.0))
            if (idx := self.tick_direction_combo.findText(s.get("tick_direction", "朝外"))) != -1: self.tick_direction_combo.setCurrentIndex(idx)
            self.minor_x_interval_spinbox.setValue(s.get("minor_x_interval", 0.5)); self.minor_y_interval_spinbox.setValue(s.get("minor_y_interval", 0.5)); self.minor_tick_length_spinbox.setValue(s.get("minor_tick_length", 4.0))
            if SCIPY_AVAILABLE: self.smooth_line_checkbox.setChecked(s.get("smooth_line", False))
            self.major_tick_length_spinbox.setValue(s.get("major_tick_length", 3.5)); self.major_tick_width_spinbox.setValue(s.get("major_tick_width", 0.8)); self.minor_tick_width_spinbox.setValue(s.get("minor_tick_width", 0.6))
        finally:
            self.is_updating_ui = False; self.update_button_color(); self.toggle_plot_settings(); self.update_plot()
        
    def create_collapsible_container(self, title, content_layout, obj_name=None):
        container = QFrame(); container.setFrameShape(QFrame.StyledPanel); container.setFrameShadow(QFrame.Raised)
        container_layout = QVBoxLayout(container); container_layout.setContentsMargins(0, 0, 0, 0)
        title_button = QPushButton(f"▼ {title}"); title_button.setStyleSheet("QPushButton { background-color: #f0f0f0; border: 1px solid #c0c0c0; border-radius: 4px; padding: 8px; text-align: left; font-weight: bold; color: #000000; } QPushButton:hover { background-color: #e0e0e0; }")
        content_widget = QWidget(); content_widget.setLayout(content_layout); content_widget.setVisible(True)
        def toggle(): content_widget.setVisible(not content_widget.isVisible()); title_button.setText(f"{'▼' if content_widget.isVisible() else '▶'} {title}")
        title_button.clicked.connect(toggle)
        container_layout.addWidget(title_button); container_layout.addWidget(content_widget)
        if obj_name: setattr(self, obj_name, container)
        return container
        
    def highlight_widget(self, widget, color="#add8e6", duration=1000):
        original_style = widget.styleSheet()
        widget.setStyleSheet(f"QFrame {{ border: 2px solid {color}; border-radius: 5px; }}")
        self.highlighted_widgets.append((widget, original_style))
        QTimer.singleShot(duration, lambda: self.clear_highlight(widget, original_style))

    def clear_highlight(self, widget, original_style):
        if (widget, original_style) in self.highlighted_widgets:
            widget.setStyleSheet(original_style); self.highlighted_widgets.remove((widget, original_style))

    def clear_all_highlights(self):
        for widget, original_style in list(self.highlighted_widgets): widget.setStyleSheet(original_style)
        self.highlighted_widgets.clear()

    def table_key_press_event(self, event):
        if event.matches(QKeySequence.StandardKey.Copy): self.copy_data()
        elif event.matches(QKeySequence.StandardKey.Paste): self.paste_data()
        else: QTableWidget.keyPressEvent(self.data_table, event)
    
    def copy_data(self):
        if not (selected_ranges := self.data_table.selectedRanges()): return
        clipboard_text = '\n'.join(['\t'.join([self.data_table.item(row, col).text() if self.data_table.item(row, col) else '' for col in range(r.leftColumn(), r.rightColumn() + 1)]) for r in selected_ranges for row in range(r.topRow(), r.bottomRow() + 1)])
        if clipboard_text: QApplication.clipboard().setText(clipboard_text)

    def paste_data(self):
        text = QApplication.clipboard().text()
        if not text: return
        lines = text.strip('\n').split('\n')
        start_row, start_col = (self.data_table.currentRow(), self.data_table.currentColumn()) if self.data_table.currentRow() >= 0 else (0, 0)
        self.is_updating_table = True
        for i, line in enumerate(lines):
            row = start_row + i
            if row >= self.data_table.rowCount(): self.data_table.insertRow(row)
            fields = line.split('\t')
            for j, field in enumerate(fields):
                if (col := start_col + j) < self.data_table.columnCount(): self.data_table.setItem(row, col, QTableWidgetItem(field.strip()))
        self.is_updating_table = False; self.update_data_from_table()
        
    def filter_table(self, text):
        for row in range(self.data_table.rowCount()):
            row_visible = any(item and text.lower() in item.text().lower() for col in range(self.data_table.columnCount()) if (item := self.data_table.item(row, col)))
            self.data_table.setRowHidden(row, not row_visible)

    def save_template(self):
        filename, _ = QFileDialog.getSaveFileName(self, "儲存範本", "", "JSON 檔案 (*.json)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f: json.dump(self.get_settings(), f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "成功", "設定範本已儲存。")
            except Exception as e: QMessageBox.critical(self, "錯誤", f"無法儲存檔案：{e}")

    def load_template(self):
        filename, _ = QFileDialog.getOpenFileName(self, "載入範本", "", "JSON 檔案 (*.json)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f: self.set_settings(json.load(f))
                QMessageBox.information(self, "成功", "設定範本已載入。")
            except Exception as e: QMessageBox.critical(self, "錯誤", f"無法載入檔案：{e}")
    
    # [V3.2.1 新增] 用於啟動/停止連續移動的函式
    def start_moving_up(self):
        self.move_row_up()
        self.move_up_timer.start(100)

    def start_moving_down(self):
        self.move_row_down()
        self.move_down_timer.start(100)

    def stop_moving(self):
        self.move_up_timer.stop()
        self.move_down_timer.stop()
            
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        main_window = PlottingApp()
        main_window.show()
        sys.exit(app.exec())
    except ImportError as e:
        QMessageBox.critical(None, "啟動失敗", f"缺少必要的函式庫，請確認已安裝 PySide6, pandas, matplotlib, numpy。\n\n詳細錯誤：{e}")
    except Exception as e:
        QMessageBox.critical(None, "程式啟動錯誤", f"啟動時發生未預期的錯誤。\n\n詳細錯誤：{e}")

