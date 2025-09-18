import cv2
from screeninfo import get_monitors

def ajustar_imagem_para_tela(img):
    screen = get_monitors()[0]
    screen_res = (screen.width, screen.height)
    height, width = img.shape[:2]
    scale_width = screen_res[0] / width
    scale_height = screen_res[1] / height
    scale = min(scale_width, scale_height, 1.0)
    window_width = int(width * scale)
    window_height = int(height * scale)
    if scale < 1.0:
        img = cv2.resize(img, (window_width, window_height), interpolation=cv2.INTER_AREA)
    return img

def carregar_imagem(caminho):
    img = cv2.imread(caminho)
    if img is None:
        print(f"Erro ao carregar a imagem: {caminho}")
        exit(1)
    return img