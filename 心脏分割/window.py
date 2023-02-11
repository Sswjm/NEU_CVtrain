# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QIcon
from inference import segmentation


class Simple_Window(QWidget):
	def __init__(self):
		super(Simple_Window, self).__init__()  # 使用super函数可以实现子类使用父类的方法
		self.setWindowTitle("心室分割与心脏病分类")
		self.setWindowIcon(QIcon('NoteBook.png'))  # 设置窗口图标
		self.resize(412, 412)
		self.text_edit = QTextEdit(self)  # 实例化一个QTextEdit对象
		#self.text_edit.setText("Hello World")  # 设置编辑框初始化时显示的文本
		self.text_edit.setPlaceholderText("Please edit text here")  # 设置占位字符串
		self.text_edit.textChanged.connect(lambda: print("text is changed!"))  # 判断文本是否发生改变
		
		self.save_button = QPushButton("Run", self)
		self.clear_button = QPushButton("Clear", self)
		
		self.save_button.clicked.connect(lambda: self.button_slot(self.save_button))
		self.clear_button.clicked.connect(lambda: self.button_slot(self.clear_button))
		
		self.h_layout = QHBoxLayout()
		self.v_layout = QVBoxLayout()
		
		self.h_layout.addWidget(self.save_button)
		self.h_layout.addWidget(self.clear_button)
		self.v_layout.addWidget(self.text_edit)
		self.v_layout.addLayout(self.h_layout)
		
		self.setLayout(self.v_layout)
	
	def button_slot(self, button):
		if button == self.save_button:
			choice = QMessageBox.question(self, "Question", "Do you want to run it?", QMessageBox.Yes | QMessageBox.No)
			if choice == QMessageBox.Yes:
				# with open('notebook.txt', 'w') as f:
					# f.write(self.text_edit.toPlainText())
				segmentation(test_path=self.text_edit.toPlainText(), save_path='result')
				self.close()
			elif choice == QMessageBox.No:
				self.close()
		elif button == self.clear_button:
			self.text_edit.clear()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Simple_Window()
	window.show()
	sys.exit(app.exec())