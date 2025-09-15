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

def main():
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    imagens = {
        "Davi": "ImagensTeste/GithubFoto.jpg",
        "Charles Leclerc": "ImagensTeste/CharlesLeclerc.png",
        "Zendaya": "ImagensTeste/Zendaya.png",
        "Ye": "ImagensTeste/Ye.png"
    }

   
    nome = "Davi"
    img = carregar_imagem(imagens[nome])
    img = ajustar_imagem_para_tela(img)

    while True:
        frame = img.copy()
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Reconhecimento Facial - by Davi Falcao", frame)
        key = cv2.waitKey(1)
        
        if cv2.getWindowProperty("Reconhecimento Facial - by Davi Falcao", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()