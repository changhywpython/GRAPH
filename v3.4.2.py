import sys
import pandas as pd
import multiprocessing
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenuBar, QMessageBox, QDialog, QLineEdit, QFormLayout,
    QDialogButtonBox, QComboBox, QSplitter, QCheckBox, QHBoxLayout,
    QLabel, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import make_interp_spline
import numpy as np

# 多核心處理的計算函式 (Worker Function)
def calculate_moving_average_chunk(args):
    """
    計算單一數據塊的移動平均。
    此函式將被每個獨立的處理程序執行。
    """
    series, window_size = args
    return series.rolling(window=window_size, min_periods=1).mean()

class CSVLoader(QThread):
    """
    使用獨立執行緒來載入 CSV/Excel 檔案，避免 GUI 卡頓。
    """
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            self.progress.emit(20)
            if self.filename.endswith('.csv'):
                df = pd.read_csv(self.filename)
            else:
                df = pd.read_excel(self.filename)
            self.progress.emit(80)
            self.finished.emit(df)
            self.progress.emit(100)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Plotter v3.4.1 - Multiprocess Enabled")
        self.setGeometry(100, 100, 1200, 800)

        self.df = None
        self.raw_df = None
        self.is_smoothed = False

        # 主佈局
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        # Matplotlib 圖表
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        splitter.addWidget(self.table)
        splitter.setSizes([600, 200])

        # 控制按鈕佈局
        control_layout = QHBoxLayout()
        self.btn_open = QPushButton("開啟檔案")
        self.btn_plot = QPushButton("繪製圖表")
        self.btn_smooth = QPushButton("平滑曲線")
        self.btn_reset = QPushButton("重置圖表")
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_plot)
        control_layout.addWidget(self.btn_smooth)
        control_layout.addWidget(self.btn_reset)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(splitter)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 連接訊號
        self.btn_open.clicked.connect(self.open_file)
        self.btn_plot.clicked.connect(self.plot_graph)
        self.btn_smooth.clicked.connect(self.smoothing_average)
        self.btn_reset.clicked.connect(self.reset_graph)

        self._create_menu()

    def _create_menu(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("檔案")
        exit_action = QAction("離開", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "開啟 CSV/Excel 檔案", "", "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx)")
        if filename:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.loader = CSVLoader(filename)
            self.loader.finished.connect(self.on_load_finished)
            self.loader.error.connect(self.on_load_error)
            self.loader.progress.connect(self.update_progress)
            self.loader.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_load_finished(self, df):
        self.raw_df = df.copy()
        self.df = df
        self.is_smoothed = False
        self.populate_table()
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "成功", "檔案載入成功！")

    def on_load_error(self, error_message):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "錯誤", f"載入檔案失敗: {error_message}")

    def populate_table(self):
        if self.df is None:
            return
        
        self.table.setRowCount(self.df.shape[0])
        self.table.setColumnCount(self.df.shape[1])
        self.table.setHorizontalHeaderLabels(self.df.columns)

        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(self.df.iat[i, j])))
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def plot_graph(self):
        if self.df is None:
            QMessageBox.warning(self, "警告", "請先載入檔案！")
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            x_column = self.df.columns[0]
            for col in self.df.columns[1:]:
                # 確保數據是數值類型
                x_data = pd.to_numeric(self.df[x_column], errors='coerce')
                y_data = pd.to_numeric(self.df[col], errors='coerce')
                # 移除包含 NaN 的行
                valid_data = pd.concat([x_data, y_data], axis=1).dropna()
                
                if not valid_data.empty:
                    ax.plot(valid_data.iloc[:, 0], valid_data.iloc[:, 1], label=col)
        except Exception as e:
            QMessageBox.critical(self, "繪圖錯誤", f"無法繪製圖表: {e}")
            return

        ax.set_title("Data Plot")
        ax.set_xlabel(self.df.columns[0])
        ax.set_ylabel("Values")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def smoothing_average(self):
        if self.df is None:
            QMessageBox.warning(self, "警告", "請先載入檔案！")
            return
        if self.is_smoothed:
            QMessageBox.information(self, "提示", "數據已經是平滑狀態。")
            return

        # 彈出對話框讓使用者輸入平滑係數
        dialog = QDialog(self)
        dialog.setWindowTitle("設定平滑係數")
        form_layout = QFormLayout()
        window_size_input = QLineEdit("10") # 預設值
        form_layout.addRow("移動平均窗口大小:", window_size_input)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        dialog.setLayout(layout)

        if dialog.exec():
            try:
                window_size = int(window_size_input.text())
                if window_size <= 1:
                    raise ValueError("窗口大小必須大於 1")
                
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0) # 顯示忙碌狀態

                # 使用多核心處理
                self.apply_smoothing_multiprocess(window_size)
                
            except ValueError as e:
                QMessageBox.critical(self, "輸入錯誤", f"無效的窗口大小: {e}")
            finally:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setVisible(False)


    def apply_smoothing_multiprocess(self, window_size):
        """
        使用 multiprocessing.Pool 來並行計算移動平均。
        """
        x_column = self.df.columns[0]
        y_columns = self.df.columns[1:]
        
        # 準備要並行處理的數據列
        tasks = [(self.df[col], window_size) for col in y_columns]
        
        try:
            # 建立處理程序池，自動偵測 CPU 核心數
            with multiprocessing.Pool() as pool:
                results = pool.map(calculate_moving_average_chunk, tasks)
            
            # 建立一個新的 DataFrame 來儲存平滑後的結果
            smoothed_df = pd.DataFrame()
            smoothed_df[x_column] = self.df[x_column]
            for i, col in enumerate(y_columns):
                smoothed_df[col] = results[i]

            self.df = smoothed_df
            self.is_smoothed = True
            
            # 更新表格和圖表
            self.populate_table()
            self.plot_graph()
            QMessageBox.information(self, "成功", f"數據已使用 {window_size} 點移動平均進行平滑。")

        except Exception as e:
            QMessageBox.critical(self, "處理錯誤", f"平滑計算失敗: {e}")


    def reset_graph(self):
        if self.raw_df is not None:
            self.df = self.raw_df.copy()
            self.is_smoothed = False
            self.populate_table()
            self.plot_graph()
            QMessageBox.information(self, "成功", "圖表與數據已重置。")
        else:
            QMessageBox.warning(self, "警告", "沒有可重置的數據。")

if __name__ == '__main__':
    # 在 Windows 上使用 multiprocessing 需要這個保護
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())