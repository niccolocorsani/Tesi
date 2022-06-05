import os

from termcolor import colored

from business_logic.correct_csv import specify_physical_object, get_all_starts, get_all_exits

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_lines(list_of_lines, img_name):
    nodes_csv = []
    edges_csv = []
    measures_csv = []
    flusso_num = 0
    print(colored(img_name, 'green'))
    for string_pattern in list_of_lines:

        try:
            string_pattern = string_pattern.replace('\n', '').lower()

            node_type = ''
            node_name = ''
            end = ''
            measure_of = ''
            type = 'Not-specified'  ## Se non è specificato ha questo valore
            bad_ends = []
            bad_measures = []
            elements = string_pattern.split(',')

            if '?' in string_pattern:
                raise Exception("Node has issue ????")

            if 'measu' not in string_pattern:
                if 'star' in elements[0]:  ## Nodo star non può avere end
                    node_type = 'start'
                    node_name = elements[0]
                elif 'exit' in elements[0]:  ## Nodo exit deve avere un end per forza
                    node_type = 'exit'
                    elements = string_pattern.split(',')
                    node_name = elements[0]
                    if len(elements) > 1: end = elements[1]
                    if '+' in string_pattern:  ##  exit puo avere più end
                        node_type = 'normal_multiple_end'
                        node_name = elements[0]
                        bad_ends = elements[1].split('+')
                        if len(bad_ends) == 0:  raise Exception("No ends in second element (line 235)")
                else:  ## Nodo normale con più end
                    if len(elements) == 1:
                        raise Exception("Should be a start maybe!!")
                    if '+' in elements[
                        1]:  ## TODO qui con un nodo tipo NODE:cl2 va in exception dicendo elemnts[1] out of bound
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

            node_name = node_name.replace('node:', '').replace(" ", "")
            node_name = node_name.replace('/', '-')
            end = end.replace('endof:', '').replace(" ", "")
            type = type.replace('type:', '').replace('/', '-').replace(" ", "")
            measure_of = measure_of.replace('measureof:', '').replace('/', '-').replace(" ", "")
            ends = []
            measures = []
            for end in bad_ends:
                if 'mea' in end: raise Exception("measu in end")
                ends.append(end.replace('endof:', '').replace(" ", "").replace('/', '-'))
            for measure in bad_measures:
                if '.' in measure: raise Exception(". AND NOT ,")
                measures.append(measure.replace('measureof:', '').replace('/', '-').replace(" ", ""))
            if node_name == '': raise Exception("Node name not present")
            if 'end' in node_name:  raise Exception("End in node name")
            if 'mea' in end: raise Exception("measu in end")
            if 'mea' in node_name: raise Exception("measu in node_name")
            if '.' in measure_of: raise Exception(". AND NOT ,")
            if '!' in node_name:  raise Exception("bad_node name")

            ##clean and invariant exception

            if node_type == 'normal_one_end':
                if 'xv' in node_name:  ## sotto nodo del nodo normale: valvola
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
                if end == '': raise Exception("Exit not should be at least an end of 1 node")

                flusso_num = flusso_num + 1
            if node_type == 'normal_multiple_end':
                if 'xv' in node_name:  ## sotto nodo del nodo normale: valvola
                    nodes_csv.append(node_name + ';Class;Valve;')  # Nodo normale valvola
                else:
                    nodes_csv.append(node_name + ';Class;PhysicalObject;')  # Nodo normale non valvola
                for end in ends:
                    edges_csv.append('Flusso_' + str(flusso_num) + ';Class;Flow;')
                    edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;startIn;' + end)
                    edges_csv.append('Flusso_' + str(flusso_num) + ';ObjectProperty;endIn;' + node_name)
                    flusso_num = flusso_num + 1
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



        except Exception as ex:
            print(colored(string_pattern, 'red'))
            print(ex)
            continue

    return nodes_csv, edges_csv, measures_csv


def read_from_file_and_place_in_list(path):
    lines = []
    with open(path) as f:
        for line in f.readlines():
            if line != '\n':
                lines.append(line)

    return lines


if __name__ == '__main__':

    path_name_original_images = os.listdir(ROOT_DIR + '/squares_image/')

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-starts.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-starts.txt')
    else:
        print("The file does not exist")

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-exits.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-exits.txt')
    else:
        print("The file does not exist")

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-nodes.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-nodes.txt')
    else:
        print("The file does not exist")

    if os.path.exists(ROOT_DIR + '/all_csv/' + 'all-measures.txt'):
        os.remove(ROOT_DIR + '/all_csv/' + 'all-measures.txt')
    else:
        print("The file does not exist")

    path_name_original_images = os.listdir(ROOT_DIR + '/images_txt/')
    lines = []

    for path_name in path_name_original_images:

        lines = read_from_file_and_place_in_list(ROOT_DIR + '/images_txt/' + path_name)
        nodes_csv, edges_csv, measures_csv = generate_lines(lines, path_name)

        specify_physical_object(nodes_csv)

        all_starts = []
        starts = get_all_starts(nodes_csv)
        all_starts.extend(starts)

        all_exits = []
        exits = get_all_exits(nodes_csv)
        all_exits.extend(exits)



        f5 = open(ROOT_DIR + '/all_csv/' + 'all-measures.txt', 'a')
        f5.write('\n')
        f5.write('---')
        f5.write('\n')
        f5.write('---')
        f5.write('\n')
        f5.write(path_name)
        f5.write('\n')
        f5.write('---')
        f5.write('\n')
        f5.write('---')
        f5.write('\n')
        for element in measures_csv:
            f5.write(element)
            f5.write('\n')







        f4 = open(ROOT_DIR + '/all_csv/' + 'all-nodes.txt', 'a')
        f4.write('\n')
        f4.write('---')
        f4.write('\n')
        f4.write('---')
        f4.write('\n')
        f4.write(path_name)
        f4.write('\n')
        f4.write('---')
        f4.write('\n')
        f4.write('---')
        f4.write('\n')
        for element in nodes_csv:
            f4.write(element)
            f4.write('\n')

        f2 = open(ROOT_DIR + '/all_csv/' + 'all-starts.txt', 'a')
        f2.write('\n')
        f2.write(path_name)
        f2.write('---')
        f2.write('\n')
        f2.write('---')
        f2.write('\n')
        for element in all_starts:
            f2.write(element)
            f2.write('\n')
            f2.flush()
        f2.close()

        f3 = open(ROOT_DIR + '/all_csv/' + 'all-exits.txt', 'a')
        f3.write(path_name)
        f3.write('\n')
        f3.write('---')
        f3.write('\n')
        f3.write('---')
        f3.write('\n')

        for element in all_exits:
            f3.write(element)
            f3.write('\n')
            f3.flush()
        f3.close()

        try:

            f = open(ROOT_DIR + '/all_csv/' + path_name.replace('.txt', '.csv'), 'w')

            f1 = open('/Users/nicc/all_csv/' + path_name.replace('.txt', '.csv'), 'w')

            for element in nodes_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()
            for element in edges_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()
            for element in measures_csv:
                f.write(element)
                f.write('\n')
                f1.write(element)
                f1.write('\n')
                f1.flush()
                f.flush()

            f.close()
            f1.close()
        except Exception as ex:
            print(ex)

    print('fine.........')
