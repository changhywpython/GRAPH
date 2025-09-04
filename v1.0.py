import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QRadioButton, QMessageBox,
    QColorDialog, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QSpinBox, QStyle, QStyleOptionButton, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import QColor
import os
from zipfile import BadZipFile
import json
import numpy as np
import re

class PlottingApp(QMainWindow):
    """
    主要應用程式視窗類別，包含 GUI 和所有繪圖邏輯。
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("多功能繪圖工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 設置 Matplotlib 字體以支援中文顯示
        try:
            # 嘗試尋找並使用系統中的中文字體
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
                # 如果常用字體找不到，則使用原本的模糊尋找方式作為備用
                font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                for font_path in font_paths:
                    if any(kw in os.path.basename(font_path).lower() for kw in ['simhei', 'yahei', 'pingfang', 'heiti']):
                        fm.fontManager.addfont(font_path)
                        plt.rcParams['font.sans-serif'] = os.path.basename(font_path).replace('.ttf', '')
                        break

        except Exception as e:
            print(f"警告: 設定中文字體失敗，可能會出現亂碼。錯誤訊息: {e}")
        
        plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
        
        # Matplotlib 圖形設定
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.legend = None # 用於追蹤圖例實例
        self.legend_press_event = None
        self.legend_drag_event = None
        self.legend_release_event = None

        self.excel_data = None
        self.x_data = []
        self.y_data = []
        self.colors_data = []
        self.data_source = 'manual' # 'manual' or 'file'
        self.selected_point_index = -1 # 新增：記錄被選中的點的索引
        self.is_updating_table = False # 新增：避免表格更新觸發無限迴圈

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

        self.init_ui()

        # 連結滑鼠點擊事件
        self.canvas.mpl_connect('button_press_event', self.on_point_click)

    def init_ui(self):
        """
        初始化 GUI 介面元件。
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左側控制面板
        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(350)
        
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

        excel_group = QGroupBox("檔案讀取")
        excel_layout = QGridLayout(excel_group)
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
        
        data_settings_layout.addWidget(excel_group)
        
        data_table_group = QGroupBox("數據表格")
        data_table_layout = QVBoxLayout(data_table_group)
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3) # 新增一欄用於顏色
        self.data_table.setHorizontalHeaderLabels(["X 數據", "Y 數據", "顏色"])
        self.data_table.setColumnWidth(2, 60) # 設定顏色欄位寬度
        # 修正: 直接連接訊號到更新函數
        self.data_table.itemChanged.connect(self.update_data_from_table)
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
        data_settings_layout.addWidget(data_table_group)

        # 繪圖設定分頁的內容
        plot_settings_layout = QVBoxLayout(self.plot_settings_tab)
        plot_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        plot_type_group = QGroupBox("圖表類型")
        plot_type_layout = QHBoxLayout(plot_type_group)
        self.line_radio = QRadioButton("折線圖")
        self.line_radio.setChecked(True)
        self.scatter_radio = QRadioButton("散佈圖")
        self.bar_radio = QRadioButton("長條圖")
        self.line_radio.toggled.connect(self.update_plot_with_timer)
        self.scatter_radio.toggled.connect(self.update_plot_with_timer)
        self.bar_radio.toggled.connect(self.update_plot_with_timer)
        plot_type_layout.addWidget(self.line_radio)
        plot_type_layout.addWidget(self.scatter_radio)
        plot_type_layout.addWidget(self.bar_radio)
        plot_settings_layout.addWidget(plot_type_group)

        settings_group = QGroupBox("圖表設定")
        settings_layout = QGridLayout(settings_group)
        self.title_input = QLineEdit()
        self.title_input.textChanged.connect(self.update_plot_with_timer)
        settings_layout.addWidget(QLabel("圖表標題:"), 0, 0)
        settings_layout.addWidget(self.title_input, 0, 1)

        self.x_label_input = QLineEdit()
        self.x_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
        self.x_label_input.textChanged.connect(self.update_plot_with_timer)
        settings_layout.addWidget(QLabel("X 軸標籤:"), 1, 0)
        settings_layout.addWidget(self.x_label_input, 1, 1)

        self.y_label_input = QLineEdit()
        self.y_label_input.setPlaceholderText("可留空，將預設為檔案欄位名稱")
        self.y_label_input.textChanged.connect(self.update_plot_with_timer)
        settings_layout.addWidget(QLabel("Y 軸標籤:"), 2, 0)
        settings_layout.addWidget(self.y_label_input, 2, 1)

        self.legend_input = QLineEdit()
        self.legend_input.setPlaceholderText("圖例名稱 (選填)")
        self.legend_input.textChanged.connect(self.update_plot_with_timer)
        settings_layout.addWidget(QLabel("圖例名稱:"), 3, 0)
        settings_layout.addWidget(self.legend_input, 3, 1)
        
        self.legend_size_spinbox = QSpinBox()
        self.legend_size_spinbox.setMinimum(1)
        self.legend_size_spinbox.setValue(10)
        self.legend_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        settings_layout.addWidget(QLabel("圖例文字大小:"), 4, 0)
        settings_layout.addWidget(self.legend_size_spinbox, 4, 1)
        
        # 新增數據標籤顯示選項
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

        data_label_size_layout = QHBoxLayout()
        self.data_label_size_spinbox = QSpinBox()
        self.data_label_size_spinbox.setMinimum(1)
        self.data_label_size_spinbox.setValue(10)
        self.data_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        data_label_size_layout.addWidget(QLabel("數據標籤大小:"))
        data_label_size_layout.addWidget(self.data_label_size_spinbox)
        
        settings_layout.addLayout(data_label_layout, 5, 0, 1, 2)
        settings_layout.addLayout(data_label_size_layout, 6, 0, 1, 2)
        
        interval_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("X/Y 軸間隔:"), 7, 0)
        self.x_interval_spinbox = QDoubleSpinBox()
        self.x_interval_spinbox.setMinimum(0.0)
        self.x_interval_spinbox.setMaximum(float('inf'))
        self.x_interval_spinbox.setValue(1.0)
        self.x_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        interval_layout.addWidget(self.x_interval_spinbox)
        
        self.y_interval_spinbox = QDoubleSpinBox()
        self.y_interval_spinbox.setMinimum(0.0)
        self.y_interval_spinbox.setMaximum(float('inf'))
        self.y_interval_spinbox.setValue(1.0)
        self.y_interval_spinbox.valueChanged.connect(self.update_plot_with_timer)
        interval_layout.addWidget(self.y_interval_spinbox)
        settings_layout.addLayout(interval_layout, 7, 1)
        
        decimal_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("小數點位數:"), 8, 0)
        self.x_decimal_spinbox = QSpinBox()
        self.x_decimal_spinbox.setMinimum(0)
        self.x_decimal_spinbox.setMaximum(999999999)
        self.x_decimal_spinbox.setValue(2)
        self.x_decimal_spinbox.valueChanged.connect(self.update_plot_with_timer)
        decimal_layout.addWidget(QLabel("X:"), alignment=Qt.AlignmentFlag.AlignRight)
        decimal_layout.addWidget(self.x_decimal_spinbox)
        
        self.y_decimal_spinbox = QSpinBox()
        self.y_decimal_spinbox.setMinimum(0)
        self.y_decimal_spinbox.setMaximum(999999999)
        self.y_decimal_spinbox.setValue(2)
        self.y_decimal_spinbox.valueChanged.connect(self.update_plot_with_timer)
        decimal_layout.addWidget(QLabel("Y:"), alignment=Qt.AlignmentFlag.AlignRight)
        decimal_layout.addWidget(self.y_decimal_spinbox)
        settings_layout.addLayout(decimal_layout, 8, 1)

        plot_settings_layout.addWidget(settings_group)

        style_group = QGroupBox("繪圖樣式設定")
        style_layout = QGridLayout(style_group)
        self.plot_color_btn = QPushButton("選擇顏色")
        self.plot_color_btn.clicked.connect(lambda: self.pick_color("plot"))
        style_layout.addWidget(QLabel("圖表顏色:"), 0, 0)
        style_layout.addWidget(self.plot_color_btn, 0, 1)

        self.bg_color_btn = QPushButton("選擇顏色")
        self.bg_color_btn.clicked.connect(lambda: self.pick_color("background"))
        style_layout.addWidget(QLabel("背景顏色:"), 1, 0)
        style_layout.addWidget(self.bg_color_btn, 1, 1)

        self.point_size_spinbox = QDoubleSpinBox()
        self.point_size_spinbox.setMinimum(1.0)
        self.point_size_spinbox.setMaximum(float('inf'))
        self.point_size_spinbox.setValue(10.0)
        self.point_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("點的大小:"), 2, 0)
        style_layout.addWidget(self.point_size_spinbox, 2, 1)
        
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setMinimum(0.5)
        self.line_width_spinbox.setMaximum(float('inf'))
        self.line_width_spinbox.setValue(2.0)
        self.line_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("線條寬度:"), 3, 0)
        style_layout.addWidget(self.line_width_spinbox, 3, 1)

        self.bar_width_spinbox = QDoubleSpinBox()
        self.bar_width_spinbox.setMinimum(0.01)
        self.bar_width_spinbox.setMaximum(1.0)
        self.bar_width_spinbox.setSingleStep(0.05)
        self.bar_width_spinbox.setValue(0.8)
        self.bar_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("長條圖寬度:"), 4, 0)
        style_layout.addWidget(self.bar_width_spinbox, 4, 1)
        
        self.border_width_spinbox = QDoubleSpinBox()
        self.border_width_spinbox.setMinimum(0.0)
        self.border_width_spinbox.setMaximum(float('inf'))
        self.border_width_spinbox.setValue(1.0)
        self.border_width_spinbox.setSingleStep(0.5)
        self.border_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("邊框粗度:"), 5, 0)
        style_layout.addWidget(self.border_width_spinbox, 5, 1)
        
        self.border_color_btn = QPushButton("選擇顏色")
        self.border_color_btn.clicked.connect(lambda: self.pick_color("border"))
        style_layout.addWidget(QLabel("邊框顏色:"), 6, 0)
        style_layout.addWidget(self.border_color_btn, 6, 1)

        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItems(["實線", "虛線", "點虛線", "點"])
        self.linestyle_combo.currentIndexChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("線條樣式:"), 7, 0)
        style_layout.addWidget(self.linestyle_combo, 7, 1)
        
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["圓形", "方形", "三角形", "星形", "無"])
        self.marker_combo.currentIndexChanged.connect(self.update_plot_with_timer)
        style_layout.addWidget(QLabel("標記樣式:"), 8, 0)
        style_layout.addWidget(self.marker_combo, 8, 1)
        
        plot_settings_layout.addWidget(style_group)
        
        axis_style_group = QGroupBox("座標軸樣式設定")
        axis_style_layout = QGridLayout(axis_style_group)
        
        axis_style_layout.addWidget(QLabel("X 軸標籤:"), 0, 0)
        self.x_label_size_spinbox = QSpinBox()
        self.x_label_size_spinbox.setMinimum(1)
        self.x_label_size_spinbox.setMaximum(999999999)
        self.x_label_size_spinbox.setValue(12)
        self.x_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        axis_style_layout.addWidget(QLabel("大小:"), 0, 1)
        axis_style_layout.addWidget(self.x_label_size_spinbox, 0, 2)
        self.x_label_bold_checkbox = QCheckBox("粗體")
        self.x_label_bold_checkbox.toggled.connect(self.update_plot_with_timer)
        axis_style_layout.addWidget(self.x_label_bold_checkbox, 0, 3)
        self.x_label_color_btn = QPushButton("顏色")
        self.x_label_color_btn.clicked.connect(lambda: self.pick_color("x_label"))
        axis_style_layout.addWidget(self.x_label_color_btn, 0, 4)
        
        axis_style_layout.addWidget(QLabel("Y 軸標籤:"), 1, 0)
        self.y_label_size_spinbox = QSpinBox()
        self.y_label_size_spinbox.setMinimum(1)
        self.y_label_size_spinbox.setMaximum(999999999)
        self.y_label_size_spinbox.setValue(12)
        self.y_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        axis_style_layout.addWidget(QLabel("大小:"), 1, 1)
        axis_style_layout.addWidget(self.y_label_size_spinbox, 1, 2)
        self.y_label_bold_checkbox = QCheckBox("粗體")
        self.y_label_bold_checkbox.toggled.connect(self.update_plot_with_timer)
        axis_style_layout.addWidget(self.y_label_bold_checkbox, 1, 3)
        self.y_label_color_btn = QPushButton("顏色")
        self.y_label_color_btn.clicked.connect(lambda: self.pick_color("y_label"))
        axis_style_layout.addWidget(self.y_label_color_btn, 1, 4)
        
        tick_size_layout = QHBoxLayout()
        axis_style_layout.addWidget(QLabel("X/Y 刻度大小:"), 2, 0)
        self.x_tick_label_size_spinbox = QSpinBox()
        self.x_tick_label_size_spinbox.setMinimum(1)
        self.x_tick_label_size_spinbox.setMaximum(999999999)
        self.x_tick_label_size_spinbox.setValue(10)
        self.x_tick_label_size_spinbox.valueChanged.connect(self.update_plot_with_timer)
        tick_size_layout.addWidget(QLabel("X:"), alignment=Qt.AlignmentFlag.AlignRight)
        tick_size_layout.addWidget(self.x_tick_label_size_spinbox)
        self.y_tick_label_size_spinbox = QSpinBox()
        self.y_tick_label_size_spinbox.setMinimum(1)
        self.y_tick_label_size_spinbox.setMaximum(999999999)
        self.y_tick_label_size_spinbox.setValue(10)
        tick_size_layout.addWidget(QLabel("Y:"), alignment=Qt.AlignmentFlag.AlignRight)
        tick_size_layout.addWidget(self.y_tick_label_size_spinbox)
        axis_style_layout.addLayout(tick_size_layout, 2, 1, 1, 4)

        self.axis_border_width_spinbox = QDoubleSpinBox()
        self.axis_border_width_spinbox.setMinimum(0.0)
        self.axis_border_width_spinbox.setMaximum(float('inf'))
        self.axis_border_width_spinbox.setValue(1.0)
        self.axis_border_width_spinbox.setSingleStep(0.5)
        self.axis_border_width_spinbox.valueChanged.connect(self.update_plot_with_timer)
        axis_style_layout.addWidget(QLabel("座標軸邊框粗度:"), 3, 0)
        axis_style_layout.addWidget(self.axis_border_width_spinbox, 3, 1, 1, 4)
        
        plot_settings_layout.addWidget(axis_style_group)
        
        grid_group = QGroupBox("網格線設定")
        grid_layout = QGridLayout(grid_group)
        self.major_grid_checkbox = QCheckBox("顯示主網格線")
        self.major_grid_checkbox.setChecked(True)
        self.major_grid_checkbox.toggled.connect(self.update_plot_with_timer)
        grid_layout.addWidget(self.major_grid_checkbox, 0, 0)
        self.minor_grid_checkbox = QCheckBox("顯示次網格線")
        self.minor_grid_checkbox.setChecked(False)
        self.minor_grid_checkbox.toggled.connect(self.update_plot_with_timer)
        grid_layout.addWidget(self.minor_grid_checkbox, 0, 1)
        self.major_grid_color_btn = QPushButton("主網格顏色")
        self.major_grid_color_btn.clicked.connect(lambda: self.pick_color("major_grid"))
        grid_layout.addWidget(self.major_grid_color_btn, 1, 0)
        self.minor_grid_color_btn = QPushButton("次網格顏色")
        self.minor_grid_color_btn.clicked.connect(lambda: self.pick_color("minor_grid"))
        grid_layout.addWidget(self.minor_grid_color_btn, 1, 1)
        plot_settings_layout.addWidget(grid_group)

        template_group = QGroupBox("範本功能")
        template_layout = QHBoxLayout(template_group)
        self.save_template_btn = QPushButton("儲存設定為範本")
        self.save_template_btn.clicked.connect(self.save_template)
        template_layout.addWidget(self.save_template_btn)
        self.load_template_btn = QPushButton("載入範本")
        self.load_template_btn.clicked.connect(self.load_template)
        template_layout.addWidget(self.load_template_btn)
        plot_settings_layout.addWidget(template_group)
        
        clear_button = QPushButton("清空圖表")
        clear_button.clicked.connect(self.clear_plot)
        clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; font-weight: bold;")
        plot_settings_layout.addWidget(clear_button)
        
        self.update_button_color()
        self.update_plot()
        self.update_table()

    def on_point_click(self, event):
        """
        處理圖表上的滑鼠點擊事件。
        """
        if event.button == 1 and event.inaxes: # 檢查是否為左鍵點擊且在圖表內
            
            # 找到圖表中的散點圖或線圖
            if self.scatter_radio.isChecked() or self.line_radio.isChecked():
                # 遍歷所有的子圖
                for artist in self.ax.get_children():
                    if isinstance(artist, plt.matplotlib.collections.PathCollection):
                        # 檢查點擊是否在集合中的點附近
                        contains, _ = artist.contains(event)
                        if contains:
                            # 找到被點選的點的索引
                            ind = _.get("ind")[0]
                            self.selected_point_index = ind
                            
                            # 更新左側面板的顏色選擇器為該點的顏色
                            point_color = self.colors_data[ind]
                            self.plot_color_hex = point_color
                            self.update_button_color()
                            
                            # 在控制台輸出訊息
                            print(f"選中了點: (X: {self.x_data[ind]}, Y: {self.y_data[ind]})")
                            return

            # 如果沒有點被選中，重置選中狀態
            self.selected_point_index = -1

    def update_button_color(self):
        """
        更新顏色選擇按鈕的背景色以反映當前顏色。
        """
        self.plot_color_btn.setStyleSheet(f"background-color: {self.plot_color_hex};")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color_hex};")
        # 更正變數名稱
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
                # 如果有選中的點，則只更改該點的顏色
                if self.selected_point_index != -1:
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
        if col == 2: # 顏色欄位
            color = QColorDialog.getColor()
            if color.isValid():
                hex_color = color.name()
                item = QTableWidgetItem(hex_color)
                # 設置背景色以視覺化顏色
                item.setBackground(QColor(hex_color))
                # 修正: 使用is_updating_table旗標來避免遞迴
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
        修正: 使用is_updating_table旗標來避免無限迴圈
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

            x_value = x_item.text() if x_item else ""
            y_value = y_item.text() if y_item else ""
            color_value = color_item.text() if color_item else self.plot_color_hex
            
            try:
                # 嘗試將字串轉換為浮點數
                new_x_data.append(float(x_value))
            except (ValueError, TypeError):
                # 如果無法轉換，則保留原始字串
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
            # 檔案讀取時，預設所有點的顏色與圖表顏色一致
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
            self.ax.set_title(f"選擇的欄位有問題: {e}", color="red")
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
                
                # 自動選擇前兩欄作為 X 和 Y
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

        if not x_to_plot or not y_to_plot:
            self.ax.set_title("請輸入或選擇數據以繪製圖表")
            self.canvas.draw()
            return

        linestyle_map = {"實線": "-", "虛線": "--", "點虛線": "-.", "點": ":"}
        marker_map = {"圓形": "o", "方形": "s", "三角形": "^", "星形": "*", "無": "None"}
        selected_linestyle = linestyle_map.get(self.linestyle_combo.currentText(), '-')
        selected_marker = marker_map.get(self.marker_combo.currentText(), 'o')
        legend_label = self.legend_input.text()

        # 處理顏色列表長度與數據不匹配的問題
        if len(colors_to_plot) != len(x_to_plot):
            colors_to_plot = [self.plot_color_hex] * len(x_to_plot)
            self.colors_data = colors_to_plot

        if self.line_radio.isChecked():
            plot_type = "折線圖"
            # 只繪製線條，不帶預設標記，避免與散點圖重疊
            self.ax.plot(x_to_plot, y_to_plot,
                         linestyle=selected_linestyle,
                         color=self.plot_color_hex,
                         linewidth=self.line_width_spinbox.value(),
                         marker='None',  # 確保 plt.plot 不繪製任何標記
                         zorder=1)
            # 獨立繪製散點，允許每個點有獨立的顏色與邊框
            self.ax.scatter(x_to_plot, y_to_plot, s=self.point_size_spinbox.value() * 1.5,
                            c=colors_to_plot,
                            edgecolors=self.border_color_hex,
                            linewidths=self.border_width_spinbox.value(),
                            marker=selected_marker, zorder=2)

        elif self.scatter_radio.isChecked():
            plot_type = "散佈圖"
            self.ax.scatter(x_to_plot, y_to_plot,
                            s=self.point_size_spinbox.value(),
                            marker=selected_marker,
                            c=colors_to_plot,
                            edgecolors=self.border_color_hex,
                            linewidths=self.border_width_spinbox.value(),
                            zorder=2)
        elif self.bar_radio.isChecked():
            plot_type = "長條圖"
            self.ax.bar(x_to_plot, y_to_plot,
                        width=self.bar_width_spinbox.value(),
                        color=colors_to_plot,
                        edgecolor=self.border_color_hex,
                        linewidth=self.border_width_spinbox.value(),
                        zorder=2)
        
        # 數據標籤顯示邏輯
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

        self.ax.set_title(self.title_input.text() or plot_type)
        
        x_label_text = self.x_label_input.text()
        y_label_text = self.y_label_input.text()
        
        self.ax.set_xlabel(x_label_text, color=self.x_label_color_hex,
                           fontweight='bold' if self.x_label_bold_checkbox.isChecked() else 'normal',
                           fontsize=self.x_label_size_spinbox.value())
        self.ax.set_ylabel(y_label_text, color=self.y_label_color_hex,
                           fontweight='bold' if self.y_label_bold_checkbox.isChecked() else 'normal',
                           fontsize=self.y_label_size_spinbox.value())
        
        self.ax.tick_params(axis='x', labelsize=self.x_tick_label_size_spinbox.value())
        self.ax.tick_params(axis='y', labelsize=self.y_tick_label_size_spinbox.value())

        # 檢查軸間隔設定，如果為 0 或空，則不設定
        if self.x_interval_spinbox.value() > 0:
            self.ax.xaxis.set_major_locator(ticker.MultipleLocator(self.x_interval_spinbox.value()))
        if self.y_interval_spinbox.value() > 0:
            self.ax.yaxis.set_major_locator(ticker.MultipleLocator(self.y_interval_spinbox.value()))

        # 設定座標軸邊框粗度
        for spine in self.ax.spines.values():
            spine.set_linewidth(self.axis_border_width_spinbox.value())
            
        self.ax.grid(self.major_grid_checkbox.isChecked(), which='major', color=self.major_grid_color_hex, linestyle='-', linewidth=0.5)
        self.ax.grid(self.minor_grid_checkbox.isChecked(), which='minor', color=self.minor_grid_color_hex, linestyle=':', linewidth=0.5)
        self.ax.minorticks_on()
        
        # 處理圖例
        if legend_label:
            # 確保只有一個圖例
            lines = self.ax.lines
            # 如果是散點圖，則需要單獨處理 legend_elements
            if not lines and self.scatter_radio.isChecked():
                handles, labels = self.ax.get_legend_elements()
                self.legend = self.ax.legend(handles, labels, prop={'size': self.legend_size_spinbox.value()}, draggable=True)
            elif lines:
                lines[0].set_label(legend_label)
                self.legend = self.ax.legend(prop={'size': self.legend_size_spinbox.value()}, draggable=True)
            
            if self.legend:
                self.legend.set_title(legend_label)
        else:
            if self.legend:
                self.legend.remove()
                self.legend = None
                
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_table(self):
        """
        根據當前數據更新表格。
        修正: 使用is_updating_table旗標來避免遞迴
        """
        self.is_updating_table = True
        
        row_count = len(self.x_data)
        self.data_table.setRowCount(row_count)
        
        for i in range(row_count):
            x_val = str(self.x_data[i])
            y_val = str(self.y_data[i])
            color_val = self.colors_data[i]

            x_item = QTableWidgetItem(x_val)
            y_item = QTableWidgetItem(y_val)
            color_item = QTableWidgetItem(color_val)
            
            # 設定顏色單元格的背景色
            color_item.setBackground(QColor(color_val))

            self.data_table.setItem(i, 0, x_item)
            self.data_table.setItem(i, 1, y_item)
            self.data_table.setItem(i, 2, color_item)
            
            # 連結顏色欄位的點擊事件
            self.data_table.cellClicked.connect(self.pick_color_for_cell)

        self.is_updating_table = False
            
    def add_row(self):
        """
        在表格中新增一行。
        """
        row_position = self.data_table.rowCount()
        self.data_table.insertRow(row_position)
        self.x_data.append(0)
        self.y_data.append(0)
        self.colors_data.append(self.plot_color_hex)
        self.update_table()
        self.update_plot_with_timer()

    def remove_row(self):
        """
        從表格中移除選定的行。
        """
        selected_rows = sorted(set(index.row() for index in self.data_table.selectedIndexes()), reverse=True)
        if not selected_rows:
            QMessageBox.warning(self, "警告", "請選擇要刪除的行。")
            return
        
        for row in selected_rows:
            self.data_table.removeRow(row)
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
            "plot_type": self.line_radio.isChecked(),
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
            "legend_size": self.legend_size_spinbox.value()
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
        self.line_radio.setChecked(settings.get("plot_type", True))
        self.scatter_radio.setChecked(not settings.get("plot_type", True))
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
        
        self.update_button_color()
        self.update_plot_with_timer()

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
    # 這是主應用程式的入口點，它檢查是否已存在一個 QApplication 實例。
    # 如果沒有，它會建立一個新的。這個檢查有助於在某些環境中（例如
    # 互動式 shell 或測試腳本）避免重複建立應用程式實例，從而防止崩潰。
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print("QApplication already running.")

    main_window = PlottingApp()
    main_window.show()
    
    # 確保只有在新的 QApplication 實例被創建時，我們才開始事件循環。
    if QCoreApplication.instance() == app:
        sys.exit(app.exec())
