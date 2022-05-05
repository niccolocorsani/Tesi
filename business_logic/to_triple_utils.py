import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_triple(string_pattern):

    string_pattern = string_pattern.lower()
    nodes_csv = []  ## will be a list of dictionary
    edges_csv = []

    node_type = ''
    node_name = ''
    end = ''
    measure_of = ''
    type = 'Not-specified'  ## Se non è specificato ha questo valore
    bad_ends = []
    bad_measures = []
    elements = string_pattern.split(',')

    if 'p851a' in elements[0]:
        print('do')
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
            node_type = 'measure_one_end'
            node_name = elements[0]
            bad_measures = elements[0].split('+')
            if len(elements) > 2:
                type = elements[2]
        else:
            node_type = 'measure_multiple_end'
            node_name = elements[0]
            measure_of = elements[1]
            if len(elements) > 2:
                type = elements[2]  ## può non essere specificato

    ##clean and invariant exception

    flusso_num = 0
    flusso_num = flusso_num + 1
    node_name = node_name.replace('node:', '')
    end = end.replace('endof:', '')
    ends = []
    measures = []
    for end in bad_ends:
        if 'mea' in end: raise Exception("measu in end")
        ends.append(end.replace('endof:', ''))
    for measure in bad_measures:
        if 'end' in end: raise Exception("end in end")
        measures.append(measure.replace('measureof:', ''))
    if node_name == '': raise Exception("Node name not present")
    if 'end' in node_name:  raise Exception("End in node name")
    if 'mea' in end: raise Exception("measu in end")
    if 'mea' in node_name: raise Exception("measu in node_name")

    ##clean and invariant exception

    if node_type == 'normal_one_end':
        if 'xv' in node_name:
            nodes_csv.append(node_name + ';Class;Valve;')  # Nodo normale valvola
            edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
            flusso_num = flusso_num + 1

        else:
            nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
            edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
            flusso_num = flusso_num + 1

    if node_type == 'start':
        nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola

    if node_type == 'exit':
        nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
        edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
        edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
        flusso_num = flusso_num + 1
    if node_type == 'normal_multiple_end':
        nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
        for end in ends:
            edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
            edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
            flusso_num = flusso_num + 1


    ##TODO da finire le misure

    return nodes_csv, edges_csv


if __name__ == '__main__':

#
# ##25
#     nodes_csv1, edges_csv1 = to_triple('NODE:TRUCK,ENDOF:XV851_1')
#     nodes_csv2, edges_csv2 = to_triple('NODE:XV852_2,ENDOF:P851A')
#
#     nodes_csv1.extend(nodes_csv2)
#     edges_csv1.extend(edges_csv2)
#
#
#
#
#     try:
#         f = open(ROOT_DIR + '/altair_semantic.txt', 'a')
#         for element in nodes_csv1:
#             f.write(element)
#             f.write('\n')
#             f.flush()
#         for element in edges_csv1:
#             f.write(element)
#             f.write('\n')
#             f.flush()
#
#         f.close()
#     except:
#         print("I/O error({0}): {1}")
#
#
#     print('fine triple utils')
#
#
#

    ##RECUPERO CO2
    nodes_csv1, edges_csv1 = to_triple('NODE:P461B,ENDOF:S461')
    nodes_csv2, edges_csv2 = to_triple('NODE:BA462,ENDOF:CH4START+S548')
    nodes_csv3, edges_csv3 = to_triple('NODE:FI460,MEASUREOF:CH4START,TYPE:NM3/H')


    nodes_csv1.extend(nodes_csv2)
    edges_csv1.extend(edges_csv2)
    nodes_csv1.extend(nodes_csv3)
    edges_csv1.extend(nodes_csv3)

    try:
            f = open(ROOT_DIR + '/all_csv/RECUPERO-CO2.txt', 'a')
            for element in nodes_csv1:
                f.write(element)
                f.write('\n')
                f.flush()
            for element in edges_csv1:
                f.write(element)
                f.write('\n')
                f.flush()

            f.close()

    except:
            print("I/O error({0}): {1}")
            print('fine triple utils')






