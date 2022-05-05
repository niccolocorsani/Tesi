##### TODO, qui deve correggere il csv mettendo cose del tipo:
## se nome inizia con s- allora cambia riga con s-855, Class,Tank  ecc....


def specify_physical_object(nodes_csv):
    for i in range(len(nodes_csv)):
        node_name = nodes_csv[i].split(';')[0]
        if 's' in node_name[0]  and 'star' not in node_name and 'exit' not in node_name and'PhysicalO' in nodes_csv[i]:
            node_element = node_name
            nodes_csv[i] = node_element + ';Class;Tank;'
        if 'p' in node_name[0] and 'star' not in node_name and 'exit' not in node_name and 'PhysicalO' in nodes_csv[i]:
            node_element = node_name
            nodes_csv[i] = node_element + ';Class;Pump;'
        if 'c' in node_name[0] and 'star' not in node_name and 'exit' not in node_name and 'PhysicalO' in nodes_csv[i]:
            node_element = node_name
            nodes_csv[i] = node_element + ';Class;Container;'
        if 'r-' in node_name[0] and 'star' not in node_name and 'exit' not in node_name and 'PhysicalO' in nodes_csv[i]:
            node_element = node_name
            nodes_csv[i] = node_element + ';Class;Reactor;'



def get_all_starts(nodes_csv):
    all_starts = []
    for i in range(len(nodes_csv)):
        node_name = nodes_csv[i].split(';')[0]
        if 'star' in node_name and 'PhysicalO' in nodes_csv[i]:
            all_starts.append(node_name.replace('start',''))


    return all_starts


def get_all_exits(nodes_csv):
    all_starts = []
    for i in range(len(nodes_csv)):
        node_name = nodes_csv[i].split(';')[0]
        if 'exit' in node_name and 'PhysicalO' in nodes_csv[i]:
            all_starts.append(node_name.replace('exit',''))


    return all_starts



####TODO fare algoritmo che cerca in tutti i nodeNAME SE vi è corrispondenza con nomi di exit e start

if __name__ == '__main__':
    pass
