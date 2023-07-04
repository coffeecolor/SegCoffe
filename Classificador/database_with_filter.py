import cv2
import numpy as np
import pandas as pd
import csv
import datetime
import time
from sklearn.cluster import MiniBatchKMeans

header = []
fill_header = True
divider = 4


class FColors:
    WARNING = '\033[93m'
    OKGREEN = '\033[92m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def piramidal (img, levels):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pyramids = [img_gray]
    for i in range(levels):
        pyramids.append(cv2.pyrDown(pyramids[i]))

    # Aplique o algoritmo de segmentação desejado (por exemplo, o's binarization)
    ret, thresh = cv2.threshold(pyramids[levels - 1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Expandir a segmentação para a imagem original
    for i in range(levels - 1, 0, -1):
        thresh = cv2.pyrUp(thresh)
        thresh = cv2.resize(thresh, (pyramids[i].shape[1], pyramids[i].shape[0]))
        # thresh = cv2.bitwise_and(pyramids[i], thresh)

    if thresh.shape != img.shape:
        thresh = cv2.resize(thresh, (img.shape[1], img.shape[0]))

    if img.dtype != thresh.dtype:
        img2 = cv2.convertTo(thresh, img.dtype)

    # Adicionar um quarto canal (canal alpha) à imagem segmentada
    result_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result_with_alpha[:, :, 3] = thresh
    return result_with_alpha


def deteccao_de_bacias(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remover os ruídos
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # Encontrar o marcador desconhecido e o marcador certo
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 1 * dist_transform.max(), 255, 0)
    # encontrar marcador desconhecido
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Encontrar marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    # Adicione 1 a todos os marcadores para que os marcadores desconhecidos sejam 1, 2, 3, ...
    markers = markers + 1
    # Marque o marcador desconhecido com 0
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    # cv2.imshow("Segmentacao", unknown)
    # cv2.imshow("Segmentacao", img)
    cv2.waitKey(0)

    # Testes para tirar fundo da segmentação
    # Inverter máscara de segmentação
    # inverted_mask = cv2.bitwise_not(segmentation_mask)

    # Aplicar máscara invertida à imagem original
    result = cv2.bitwise_and(img, img, mask=unknown)
    result_with_alpha2 = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_with_alpha2[:, :, 3] = unknown

    return result_with_alpha2


def limiarizacao(img, limiar):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplique a limiarização com o limiar de 64
    ret, img_threshold = cv2.threshold(img_gray, limiar, 255, cv2.THRESH_BINARY)

    # Testes para tirar fundo da segmentação
    inverted_mask = cv2.bitwise_not(img_threshold)
    # cv2.imshow("Result", inverted_mask)
    # Aplicar máscara invertida à imagem original
    result = cv2.bitwise_and(img, img, mask=inverted_mask)
    #   Adicionar um quarto canal (canal alpha) à imagem resultante
    result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # Definir a transparência para os pixels de fundo (onde a máscara binária é igual a 0)
    result_with_alpha[:, :, 3] = inverted_mask
    return result_with_alpha

def kmeans(img, k):
    # A imagem é transformada em uma matriz unidimensional de dimensão 3, onde cada linha representa um pixel e cada coluna representa um canal de cor (vermelho, verde e azul).
    Z = img.reshape((-1, 3))
    # converte para np.float32
    Z = np.float32(Z)
    # Defina as condições de parada do algoritmo K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Defina o número de grupos em que deseja dividir os pixels da imagem
    K = k
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # converte de volta para uint8
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # Criar uma máscara com as áreas segmentadas em branco e o resto em preto
    mask = np.zeros((img.shape[0] * img.shape[1], 1), np.uint8)
    majority_label = np.argmax(np.bincount(label.flatten()))
    mask[label.flatten() == majority_label] = 255
    mask = mask.reshape((img.shape[0], img.shape[1]))

    # Aplicar a máscara na imagem original
    inv_mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=inv_mask)

    #   Adicionar um quarto canal (canal alpha) à imagem resultante
    result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    # Definir a transparência para os pixels de fundo (onde a máscara binária é igual a 0)
    result_with_alpha[:, :, 3] = inv_mask

    return result_with_alpha


def region_growing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(img_gray)

    queue = []
    seed = (img_gray.shape[0] // 2, img_gray.shape[1] // 2)
    queue.append(seed)

    # define a tolerância de cor
    tolerance = 70

    # laço até que a fila esteja vazia
    while len(queue) > 0:
        pixel = queue.pop(0)

        # checar se o pixel está dentro da imagem
        if (pixel[0] >= 0 and pixel[0] < img_gray.shape[0] and
                pixel[1] >= 0 and pixel[1] < img_gray.shape[1]):
            # Checar se o pixel já foi processado
            if output[pixel[0], pixel[1]] == 0:
                # checar se o pixel está dentro da tolerância
                if abs(int(img_gray[pixel[0], pixel[1]]) - int(img_gray[seed[0], seed[1]])) < tolerance:
                    output[pixel[0], pixel[1]] = 255

                    # adicionar os pixels vizinhos na fila
                    queue.append([pixel[0] - 1, pixel[1]])
                    queue.append([pixel[0] + 1, pixel[1]])
                    queue.append([pixel[0], pixel[1] - 1])
                    queue.append([pixel[0], pixel[1] + 1])

    # inverter saida
    output = 255 - output

    inverted_mask = cv2.bitwise_not(output)
    result = cv2.bitwise_and(img, img, mask=inverted_mask)

    result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_with_alpha[:, :, 3] = inverted_mask

    return result_with_alpha


def crop_image(img, x, y, h):
    return img[y:y+h, x:x+h]


def extract_hsv(img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    return (H, S, V)

def extract_lab(img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    return (L, A, B)

def extract_grey(img):
    bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    grey_img = cv2.cvtColor(bgra_img, cv2.COLOR_BGRA2GRAY)
    return grey_img


def mean_std(data, color_space, type):
    list_data = []
    for index, component in enumerate(data):
        if component.shape[-1] == 4:
            continue  # ignore the alpha channel
        list_data.append(np.mean(component))
        header.append('{}_{}_mean'.format(type, color_space[index])) if fill_header else None
        list_data.append(np.std(component))
        header.append('{}_{}_std'.format(type, color_space[index])) if fill_header else None

    return list_data


def equalize_hist(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Dividindo a imagem HSV em diferentes canais
    h, s, v = cv2.split(hsv_img)

    # Aplicando equalização de histograma no canal V
    v_equalized = cv2.equalizeHist(v)

    # Unificando os canais H, S e V com equalização aplicada
    hsv_img_equalized = cv2.merge((h, s, v_equalized))

    # Convertendo imagem HSV equalizada em RGB
    img = cv2.cvtColor(hsv_img_equalized, cv2.COLOR_HSV2BGR)
    return img


def color_quantization(img, n_clusters=6):
    # Pega tamanho da imagem
    (h, w) = img.shape[:2]

    # converte a imagem do espaço de cores RGB para o espaço de cores L*a*b*
    # -- já que estaremos agrupando usando k-means # que é baseado na distância euclidiana, usaremos o
    # L*a* b* espaço de cor onde a distância euclidiana implica
    # significado perceptivo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # remodela a imagem em um vetor para que o k-means possa ser aplicado
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Aplica o KMeans
    clt = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = clt.fit_predict(img)

    # A imagem reduzida a cores
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # remodela o vetor para imagem novamente
    quant = quant.reshape((h, w, 3))

    # converte de L*a*b* para RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    # Retorna imagem com cores reduzidas
    return quant


def get_image_paths():
    global fill_header
    data = []
    dataframe = pd.read_csv('./photos.csv', delimiter=';')

    for index, row in dataframe.iterrows():
        row_data = []
        #img_coffee = kmeans(cv2.imread('./RAW/{}'.format(row['name_coffee'])), 2)
        #img_coffee = deteccao_de_bacias(cv2.imread('./RAW/{}'.format(row['name_coffee'])))
        img_coffee = region_growing(cv2.imread('./RAW/{}'.format(row['name_coffee'])))
        #img_coffee = piramidal(cv2.imread('./RAW/{}'.format(row['name_coffee'])), 4)
        #img_coffee = limiarizacao(cv2.imread('./RAW/{}'.format(row['name_coffee'])), limiar)
        # img_paper = crop_image(cv2.imread(
        #     './RAW/{}'.format(row['name_paper'])), row['X1'], row['Y1'], row['H'])
        agtron_value = row['agtron']
        flash_value = row['flash']

        # Suavização pela mediana
        #img_coffee = cv2.medianBlur(src=img_coffee, ksize=5)
        # img_paper = cv2.medianBlur(src=img_paper, ksize=5)

        # Quantização de cores
        # img_coffee = color_quantization(img_coffee, n_clusters=3)
        # img_paper = color_quantization(img_paper, n_clusters=3)

        # Equalização de histograma
        # img_coffee = equalize_hist(img_coffee)
        # img_paper = equalize_hist(img_paper)

        print('device: {}, flash: {}, agtron: {}'.format(
            row['device_id'], flash_value, agtron_value))

        # Cria um dicionário com os dados da imagem de café (Componentes referentes a cor)
        row_data.extend(mean_std(extract_hsv(img_coffee),
                                 ['H', 'S', 'V'], 'coffee'))
        row_data.extend(mean_std(extract_lab(img_coffee),
                                 ['L', 'A', 'Bl'], 'coffee'))
        row_data.extend([np.mean(extract_grey(img_coffee)),
                         np.std(extract_grey(img_coffee))])
        header.extend(['coffee_grey_mean', 'coffee_grey_std']
                      ) if fill_header else None

        # # Cria um dicionário com os dados da imagem de papel (Componentes referentes a iluminação)
        # row_data.extend(
        #     mean_std([extract_hsv(img_paper)[2]], ['V'], 'paper'))
        # row_data.extend(
        #     mean_std([extract_lab(img_paper)[0]], ['L'], 'paper'))

        row_data = [round(num, 3) for num in row_data]
        row_data.extend([flash_value, 'Agtron {}'.format(agtron_value)])
        header.extend(['flash', 'agtron']) if fill_header else None

        data.append(row_data)
        fill_header = False
    return data


def export_csv(data, name):
    with open('./DATA/{}.csv'.format(name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # Escreve o cabeçalho (header)
        writer.writerow(header)
        # Escreve todas as linhas (data)
        writer.writerows(data)

        f.close()


if __name__ == '__main__':
    # ms = datetime.datetime.now()
    # with open('tempos.txt', 'w') as f:
    #     for limiar in range(130, 150):
    #         print(f"{FColors.BOLD}{FColors.WARNING}{'Limiar: '}{limiar}{FColors.ENDC}\n")
    #         loop_start_time = time.time()
    #
    #         FILE = f'Limiar_{limiar}_' + str(round(time.mktime(ms.timetuple()) * 1000))
    #         export_csv(get_image_paths(limiar), FILE)
    #
    #         loop_end_time = time.time()
    #         loop_time = loop_end_time - loop_start_time
    #         print(f"{FColors.BOLD}{FColors.WARNING}{'Tempo de execução: '}{loop_time}{' s'}{FColors.ENDC}\n")
    #
    #         f.write(f"Limiar {limiar}: {loop_time} s\n")
    #
    #     # Fecha o arquivo após a escrita
    #     f.close()
    #




    start_time = time.time()
    ms = datetime.datetime.now()

    # warnings.filterwarnings(action='ignore')
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")

    FILE = 'Crescimento_Regioes_' + str(round(time.mktime(ms.timetuple()) * 1000))
    export_csv(get_image_paths(), FILE)

    end_time = time.time()
    print(f"{FColors.BOLD}{FColors.WARNING}{'time = '}{end_time - start_time}{' s'}{FColors.ENDC}\n")
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")

