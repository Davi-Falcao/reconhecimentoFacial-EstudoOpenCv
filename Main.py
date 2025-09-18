import cv2
import TratamentoImagem

def main():
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    imagens = {
        "Davi": "ImagensTeste/GithubFoto.jpg",
        "Charles Leclerc": "ImagensTeste/CharlesLeclerc.png",
        "Zendaya": "ImagensTeste/Zendaya.png",
        "Ye": "ImagensTeste/Ye.png"
    }

    nome = "Davi"  
    
    img = TratamentoImagem.carregar_imagem(imagens[nome])
    img = TratamentoImagem.ajustar_imagem_para_tela(img)

    while True:
        frame = img.copy()
        #Convertendo para escala de cinza
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Detectando faces baseado no classificador treinado com a imagem cinza
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        #Desenhando retangulos ao redor das faces 
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Reconhecimento Facial - by Davi Falcao", frame)       
        cv2.waitKey(1) 

        #Ao fechar a janela sai do loop (As vezes o método waitKey não detecta o fechamento da janela)
        if cv2.getWindowProperty("Reconhecimento Facial - by Davi Falcao", cv2.WND_PROP_VISIBLE) < 1:
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()