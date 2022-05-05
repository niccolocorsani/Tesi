import sys
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 textbox - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)



        self.textEdit = QTextEdit(self)
        self.textEdit.move(20, 80)
        self.textEdit.resize(400, 500)


        # Create a button in the window
        self.button = QPushButton('Show text', self)
        self.button.move(20, 700)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        # QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QMessageBox.StandardButton.Ok,
        #                      QMessageBox.StandardButton.Ok)
        self.textEdit.append(self.create_triple(textboxValue))

    def create_triple(self, string_pattern):


        string_pattern = string_pattern.lower()
        nodes_csv = []
        edges_csv = []
        measures_csv = []
        added_nodes = []
        added_ends = []

        node_type = ''
        node_name = ''
        end = ''
        measure_of = ''
        type = 'Not-specified'  ## Se non è specificato ha questo valore
        bad_ends = []
        bad_measures = []
        elements = string_pattern.split(',')

        flusso_num = 0


        if '?' in string_pattern:
            raise Exception("Node has issue ????")
        if ':' not in string_pattern:
            raise Exception("Node doesn'contain :")
        if 'measu' not in string_pattern:
            if 'star' in elements[0]:  ## Nodo star non può avere end
                node_type = 'start'
                node_name = elements[0]
            elif 'exit' in elements[0]:  ## Nodo exit deve avere un end per forza
                node_type = 'exit'
                elements = string_pattern.split(',')
                node_name = elements[0]
                end = elements[1]
            else:  ## Nodo normale con più end
                if '+' in elements[1]:
                    node_type = 'normal_multiple_end'
                    node_name = elements[0]
                    bad_ends = elements[1].split('+')
                    if len(bad_ends) == 0:  raise Exception("No ends in second element (line 235)")
                else:  ## Nodo normale con un end
                    node_type = 'normal_one_end'
                    node_name = elements[0]
                    end = elements[1]

        if 'measu' in string_pattern:  ## Nodo measure
            if '+' in elements[1]:
                node_type = 'measure_multiple'
                node_name = elements[0]
                bad_measures = elements[1].split('+')
                if len(elements) > 2:
                    type = elements[2]
            else:
                node_type = 'measure_one'
                node_name = elements[0]
                measure_of = elements[1]
                if len(elements) > 2:
                    type = elements[2]  ## può non essere specificato

        ##clean and invariant exception

        node_name = node_name.replace('node:', '')
        end = end.replace('endof:', '')
        type = type.replace('type:', '')
        measure_of = measure_of.replace('measureof:', '')
        ends = []
        measures = []
        for end in bad_ends:
            if 'mea' in end: raise Exception("measu in end")
            ends.append(end.replace('endof:', ''))
        for measure in bad_measures:
            if '.' in measure: raise Exception(". AND NOT ,")
            measures.append(measure.replace('measureof:', ''))
        if node_name == '': raise Exception("Node name not present")
        if 'end' in node_name:  raise Exception("End in node name")
        if 'mea' in end: raise Exception("measu in end")
        if 'mea' in node_name: raise Exception("measu in node_name")
        if '.' in measure_of: raise Exception(". AND NOT ,")

        ##clean and invariant exception

        ##TODO mettere che se esiste un end di qualcosa che non è stato definito precedentemnete fa raise exception

        if node_type == 'normal_one_end':
            if 'xv' in node_name:  ## sotto nodo del nodo normale: valvola
                nodes_csv.append(node_name + ';Class;Valve;')  # Nodo normale valvola
                edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                flusso_num = flusso_num + 1
                ##TO debug
                added_nodes.append(node_name)
                added_ends.append(end)
                ##To debug
            else:
                nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
                edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                flusso_num = flusso_num + 1
                ##TO debug
                added_nodes.append(node_name)
                added_ends.append(end)
                ##To debug
        if node_type == 'start':
            nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
            ##TO debug
            added_nodes.append(node_name)
            added_ends.append(end)
            ##To debug
        if node_type == 'exit':
            nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
            edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
            ##TO Debug
            added_ends.append(end)
            ##TO Debug
            flusso_num = flusso_num + 1
        if node_type == 'normal_multiple_end':
            nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
            ##TO debug
            added_nodes.append(node_name)
            added_ends.append(end)
            ##To debug
            for end in ends:
                edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                flusso_num = flusso_num + 1
                ##TO debug
                added_ends.append(end)

                ##TO debug

            ##TO debug
            added_nodes.append(node_name)
            ##TO debug

        if node_type == 'measure_one':
            if '%' in type:
                type = 'percentage'
            measures_csv.append(node_name + ';Class;Sensor;')
            measures_csv.append(type + ';Class;UnitOfMeasure;')

            measures_csv.append(node_name + ';ObjectProperty;canMeasureIn;' + type)
            measures_csv.append(node_name + ';ObjectProperty;isContainedIn;' + measure_of)

        if node_type == 'measure_multiple':
            if '%' in type:
                type = 'percentage'
            measures_csv.append(node_name + ';Class;Sensor;')
            measures_csv.append(type + ';Class;UnitOfMeasure;')
            measures_csv.append(node_name + ';ObjectProperty;canMeasureIn;' + type)
            for measure in measures:
                measures_csv.append(node_name + ';ObjectProperty;isContainedIn;' + measure)

        all_string = ''

        for string in nodes_csv:
            all_string = all_string + string + '\n'

        for string in edges_csv:
            all_string = all_string + string + '\n'

        for string in measures_csv:
            all_string = all_string + string + '\n'

        return all_string



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec())

    #### sembra non funzioni su m1: have 'x86_64', need 'arm64e'
