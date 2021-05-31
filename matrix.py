import numpy as np
from scipy.sparse import coo_matrix
from tensorflow.keras.layers import Dense


def matrix2coords(array, sign_edge_direccion=False, offsetSrc=0, offsetDst=0):
    it = np.nditer(array, flags=['multi_index'])
    values = []
    rows = []
    cols = []
    while not it.finished:
        # print("%1.4f <%s>" % (float(it[0]), it.multi_index), end=' ')
        value = float(it[0])
        if sign_edge_direccion:
            if value > 0:
                rows.append(it.multi_index[0] + offsetSrc)
                cols.append(it.multi_index[1] + offsetDst)
                values.append(value)
            else:
                cols.append(it.multi_index[0] + offsetSrc)
                rows.append(it.multi_index[1] + offsetDst)
                values.append(abs(value))
        else:
            rows.append(it.multi_index[0] + offsetSrc)
            cols.append(it.multi_index[1] + offsetDst)
            values.append(value)

        is_not_finished = it.iternext()
    return rows, cols, values


def model2matrix(model):
    layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    # Build distance matrix
    rows = list()
    cols = list()
    values = list()
    offset = 0
    for idx, layer in enumerate(layers):
        weights, weights_bias = layer.get_weights()
        r, c, v = matrix2coords(weights, offsetSrc=offset, offsetDst=offset + len(weights))
        offset = offset + len(weights)
        rows.append(r)
        cols.append(c)
        values.append(v)

    rows_total = np.concatenate(rows)
    cols_total = np.concatenate(cols)
    values_total = np.concatenate(values)

    print('rows: ' + str(len(rows_total)) + ', cols: ' + str(len(cols_total)) + ', values_total: ' + str(
        len(values_total)))

    max_rows = np.amax(rows_total) + 1
    max_cols = np.amax(cols_total) + 1
    max_dim = max(max_rows, max_cols)
    print('max_rows: ' + str(max_rows) + ', max_cols: ' + str(max_cols) + ', max_dim: ' + str(max_dim))

    min_value = np.amin(values_total)
    max_value = np.amax(values_total)
    print('min_value: ' + str(min_value) + ', max_value: ' + str(max_value))
    print('Graph num edges: ' + str(len(values_total)))

    # ojo debe tener el sentido  de una distancia
    # normalized_values_total = values_total - min_value  # no es una distancia
    # normalized_values_total = max_value - values_total + 0.0001 #normalized_values_total queremos 0s que no esten en la
    min_edge_distance = 0.0001

    normalized_values_total = min_edge_distance + max(abs(max_value), abs(min_value)) - np.absolute(values_total)
    # norm_min_value = np.amin(normalized_values_total)

    with open('./output/mlp_distance_matrix.flag', 'w') as file:
        file.write('dim 0:\n')
        strNodes = ''
        for pos in list(range(max_dim)):
            strNodes = strNodes + '0 '
        strNodes = strNodes.strip()
        file.write(strNodes + '\n')

        file.write('dim 1:\n')
        for pos in list(range(len(rows_total))):
            file.write(str(rows_total[pos]) + ' ' + str(cols_total[pos]) + ' ' +
                       '{:.5f}'.format(normalized_values_total[pos]) + '\n')

    distance_matrix_coo = coo_matrix((normalized_values_total, (rows_total, cols_total)), shape=(max_dim, max_dim))
    distance_matrix = (distance_matrix_coo + distance_matrix_coo.T) / 2  # simetrize

    distance_matrix_dense = distance_matrix.toarray().transpose(1, 0)
    distance_matrix_dense = np.where(distance_matrix_dense < min_edge_distance, np.inf, distance_matrix_dense)
    np.fill_diagonal(distance_matrix_dense, 0)

    return distance_matrix_coo, distance_matrix_dense
