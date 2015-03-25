#!/usr/bin/env python3

import os.path

from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidget

from sklearn.cluster import KMeans

### custom imports follow ###
from reppar import RulesParser
from tabpar import TabDataParser
from procrules import ProcRules

class DiplomMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi(os.path.join("ui", "mainw.ui"), self)

        self.setWindowTitle("Кластеризация логических закономерностей")

        self.tabWidget.setTabText(0, "Пареметры")
        self.tabWidget.setTabText(1, "Результаты")

        self.gb_datasrc.setTitle("Источники данных")
        self.freport = os.path.join("..", "data", "wine-lrules.html")
        self.lbl_report.setText("Файл отчета")
        self.le_repfname.setPlaceholderText(self.freport)
        self.pb_report.setText("Обзор")
        self.pb_report.clicked.connect(self.onReportClicked)

        self.ftrain = os.path.join("..", "data", "wine-train.tab")
        self.lbl_train.setText("Обучающая выборка")
        self.le_trainfname.setPlaceholderText(self.ftrain)
        self.pb_train.setText("Обзор")
        self.pb_train.clicked.connect(self.onTrainClicked)

        self.ftest = os.path.join("..", "data", "wine-test.tab")
        self.lbl_test.setText("Контрольная выборка")
        self.le_testfname.setPlaceholderText(self.ftest)
        self.pb_test.setText("Обзор")
        self.pb_test.clicked.connect(self.onTestClicked)

        self.gb_settings.setTitle("Параметры кластеризации")
        self.lbl_class.setText("Класс")
        self.cb_class.setEnabled(False)

        self.nclusters = 2
        self.sb_clusters.valueChanged[int].connect(
            self.onClustersValueChanged
        )
        self.sb_clusters.setMinimum(2)
        self.lbl_clusters.setText("кластеров")

        self.pb_cluster.setText("Кластеризовать")
        self.pb_cluster.clicked.connect(self.onClusterClicked)

        self.onReportClickedGo()
        self.onTrainClickedGo()
        self.onTestClickedGo()

    def run_in_thread(self, thread, worker, slot_good, slot_bad):
        worker.moveToThread(thread)
        worker.done.connect(slot_good)
        worker.failed.connect(slot_bad)

        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.started.connect(worker.work)
        thread.start()

    def onReportClicked(self):
        self.freport = QFileDialog.getOpenFileName()[0]
        self.le_repfname.setText(self.freport)

        self.onReportClickedGo()

    def onReportClickedGo(self):
        self.pb_report.setEnabled(False)
        self.report_thread = QThread()
        self.report_worker = RulesWorker(self.freport)
        self.run_in_thread(
            self.report_thread, self.report_worker,
            self.onReportParsed, self.onReportFailed
        )

    @pyqtSlot(RulesParser)
    def onReportParsed(self, report_parser):
        self.report_parser = report_parser
        self.le_repfname.setText(self.freport)
        self.pb_report.setEnabled(True)

        self.cb_class.addItems(
            [str(k) for k in report_parser.rules.keys()]
        )
        self.cb_class.setEnabled(True)

    @pyqtSlot()
    def onReportFailed(self):
        self.report_parser = None
        self.le_repfname.setText("")
        self.pb_report.setEnabled(True)

        self.cb_class.clear()
        self.cb_class.setEnabled(False)

    def onTrainClicked(self):
        self.ftrain = QFileDialog.getOpenFileName()[0]
        self.le_trainfname.setText(self.ftrain)

        self.onTrainClickedGo()

    def onTrainClickedGo(self):
        self.pb_train.setEnabled(False)
        self.train_thread = QThread()
        self.train_worker = TabDataWorker(self.ftrain)
        self.run_in_thread(
            self.train_thread, self.train_worker,
            self.onTrainParsed, self.onTrainFailed
        )

    @pyqtSlot(TabDataParser)
    def onTrainParsed(self, train_parser):
        self.train_parser = train_parser
        self.le_trainfname.setText(self.ftrain)
        self.pb_train.setEnabled(True)

    def onTrainFailed(self):
        self.train_parser = None
        self.le_trainfname.setText("")
        self.pb_train.setEnabled(True)

    def onTestClicked(self):
        self.ftest = QFileDialog.getOpenFileName()[0]
        self.le_testfname.setText(self.ftest)

        self.onTestClickedGo()

    def onTestClickedGo(self):
        self.pb_test.setEnabled(False)
        self.test_thread = QThread()
        self.test_worker = TabDataWorker(self.ftest)
        self.run_in_thread(
            self.test_thread, self.test_worker,
            self.onTestParsed, self.onTestFailed
        )

    @pyqtSlot(TabDataParser)
    def onTestParsed(self, test_parser):
        self.test_parser = test_parser
        self.le_testfname.setText(self.ftest)
        self.pb_test.setEnabled(True)

    def onTestFailed(self):
        self.test_parser = None
        self.le_testfname.setText("")
        self.pb_test.setEnabled(True)

    @pyqtSlot(int)
    def onClustersValueChanged(self, nclusters):
        self.nclusters = nclusters

    def onClusterClicked(self):
        pr = ProcRules(self.train_parser, self.report_parser)
        km = KMeans(n_clusters = self.nclusters)
        km.fit(pr.rules[int(self.cb_class.currentText())])

        print(km.cluster_centers_)


class RulesWorker(QObject):
    done = pyqtSignal(RulesParser)
    failed = pyqtSignal()

    def __init__(self, freport):
        super().__init__()
        self.freport = freport

    def __del__(self):
        print("ReportWorker del")

    def work(self):
        try:
            self.done.emit(RulesParser(self.freport))
        except:
            self.failed.emit()


class TabDataWorker(QObject):
    done = pyqtSignal(TabDataParser)
    failed = pyqtSignal()

    def __init__(self, fname):
        super().__init__()
        self.fname = fname

    def __del__(self):
        print("TabWorker del")

    def work(self):
        try:
            self.done.emit(TabDataParser(self.fname))
        except:
            self.failed.emit()

if __name__ == "__main__":
    import sys, logging.config
    from PyQt5.QtWidgets import QApplication

    from log import logsettings

    logging.config.dictConfig(logsettings)

    app = QApplication(sys.argv)
    dmw = DiplomMainWindow()
    dmw.show()

    sys.exit(app.exec_())
