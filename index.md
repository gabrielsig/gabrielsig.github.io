# Considerações iniciais

Esta página tem o objetivo de exibir os trabalhos realizados para a disciplina Processamento DIgital de Imagens (DCA0445) ministrada pelo professor [Agostinho Brito](http://agostinhobritojr.github.io/).

Todos os códigos foram implementados em Python com a biblioteca OpenCV. Os tutoriais, assim como os exercícios, disponibilizados pelo professor podem ser vistos [aqui](http://agostinhobritojr.github.io/tutoriais/pdi/).  

# Unidade 1
## 1. Aplicação de negativo
#### 1.1. Descrição

Neste programa o usuário deve fornecer as coordenadas de dois pontos, P1 e P2, dentro dos limites da imagem. A partir disso o programa deverá aplicar um efeito de negativo na área da imagem localizada no interior da região definida por esses pontos pontos.

A fim de tornar o programa mais interativo, decidimos usar o mouse para definir a área entre os pontos ao invés de solicitar que o usuário digite as coordenadas na linha de comando. Essa lógica foi feita da seguinte maneira: Ao clicarmos e segurarmos o botão esquerdo do mouse em algum ponto da imagem, o programa armazena as coordenadas no ponto P1; então deve-se arrastar o mouse até a posição final e soltar o botão, assim o programa armazena esse segundo conjunto de coordenadas no ponto P2.

O código completo dessa aplicação pode ser obtido [agui](github.com/gabrielsig)

#### 1.2. Explicando o código

Inicialmente carregamos uma imagem qualquer e extraímos suas dimensões. Então criamos duas listas de 2 elementos que servirão para armazenar as coordenadas dos pontos, tais como duas flags que serão usadas para checarmos se os pontos foram definidos ou não. Por último criamos a janela onde o resultado será exibido, e definimos a função que será chamada quando o mouse for clicado (mais sobre isso a frente):

```Python
# read image
img = cv2.imread('media/hermes2.jpeg')

# get the image height and width
img_height, img_width = img.shape[:2]

# create the points on the origin and on the farther edge of the image
p1 = [0, 0]
p2 = [img_height, img_width]

# flags to determine if either one of the points is set
p1_set = False
p2_set = False

# create the image and the mouse callback
cv2.namedWindow('Regions')
cv2.setMouseCallback('Regions', on_mouse_select)
```

A função `on_mouse_select` é chamada sempre que um evento relacionado ao mouse acontecer. Ela recebe como parâmetros as coordenadas do mouse e o evento que gerou o acionamento:

```Python
def on_mouse_select(event, x, y, flags, param):
    global p1, p2, p1_set, p2_set, img_width, img_height

    # if the point is outside of the image, set its coordinates to the border
    if x > img_width:
        x = img_width
    if y > img_height:
        y = img_height

    # assign the coordinates to the points and set the flags
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = [y, x]
        p1_set = True
    elif event == cv2.EVENT_LBUTTONUP:
        p2 = [y, x]
        p2_set = True
```

Por último entramos em um loop onde é verificado  se o usuário já forneceu o conjunto de pontos. Em caso positivo, o interior da região composta pelos dois pontos é varrida, o valor de intensidade de cada pixel é substituído pelo seu complemento e as flags são resetadas.

Posteriormente esperamos que o usuário pressione alguma das duas teclas: caso pressione a barra de espaço, a imagem é recarregada, caso pressione esc o programa é finalizado:

```Python
while True:
    if p1_set and p2_set:
        # Apply the negative in the area composed by the 2 points provided by the user
        for i in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
            for j in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
                img[i, j] = 255 - img[i, j]
        # reset the flags
        p1_set = False
        p2_set = False

    # show the image
    cv2.imshow('Regions', img)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close the program
    # if the space bar is pressed, reset the image
    if key == 27:
        break
    elif key == 32:
        img = cv2.imread('media/hermes2.jpeg')

cv2.destroyAllWindows()

```

#### 1.3 Resultados

Na primeira imagem podemos ver o resultado após uma região qualquer ser selecionada pelo usuário:

![regions1](https://github.com/gabrielsig/gabrielsig.github.io/images/regions/regions1.png)

Já na segunda imagem podemos ver o que acontece quando o usuário seleciona duas regiões que possuem uma interseção. Nessa região o valor dos pixels volta ao original, pois o negativo do negativo é a própria imagem:

![regions1](https://github.com/gabrielsig/gabrielsig.github.io/images/regions/regions2.png)

## 2. Troca de regiões da imagem 
