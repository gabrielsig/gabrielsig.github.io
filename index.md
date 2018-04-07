# Processamento Digital de Imagens

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

```python
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

```python
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

```python
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

![regions1](gabrielsig.github.io/images/regions/regions1.png)

Já na segunda imagem podemos ver o que acontece quando o usuário seleciona duas regiões que possuem uma interseção. Nessa região o valor dos pixels volta ao original, pois o negativo do negativo é a própria imagem:

![regions1](gabrielsig.github.io/images/regions/regions2.png)

## 2. Troca de regiões da imagem
#### 2.1. Descrição

Neste exercício foi proposto a implementação de um programa que carregue uma imagem e faça a troca dos quadrantes da mesma em diagonal. Porém, além dessa implementação, decidimos construir um programa que solicita do usuário em quantas partes ele quer dividir a imagem, realiza um embaralhamento aleatório dessas sub imagens e monta uma nova imagem.

O código completo da troca de regiões pode ser obtido [aqui]() e o do embaralhamento [aqui]()

#### 2.2. Explicando o código

Como o programa de embaralhamento é construído em cima do programa que apenas troca as regiões da imagem, usamos ele como referência nas explicações por ser um pouco mais complexo.

Inicialmente a imagem é carregada e obtemos sua altura e largura, solicitamos do usuário quantos blocos horizontais e verticais ele quer que a imagem seja dividida e checamos se os valores fornecidos são divisores das dimensões da imagem. Caso os valores sejam válidos, calculamos os tamanhos dos blocos que serão usados.

```python
img = cv2.imread('media/hermes2.jpeg')

height, width = img.shape[:2]

print('Image size: x = {} | y = {}'.format(height, width))

n_rows = int(input("Type the number of rows: "))
n_cols = int(input("Type the number of columns: "))

if height % n_rows != 0:
    print("[ERROR]: image height is not divisible by the number provided")
    exit()
elif width % n_cols != 0:
    print("[ERROR]: image height is not divisible by the number provided")
    exit()

x_size = int(height / n_rows)
y_size = int(width / n_cols)
```

Criamos uma lista (`s`) onde armazenaremos os blocos da imagem. Para gerar esses blocos, percorremos a imagem em incrementos de `x_size` de 0 até `height` e em incrementos de `y_size` de 0 até `width`. Em cada iteração desses loops aninhados, o bloco com coordenadas `(i,  j)` até `(i + x_size, j + y_size)` é concatenado no array `s`.

Com o vetor `s` preenchido, aplicamos a função `np.random.shuffle()` para realizar o embaralhamento das posições.

```python
s = []

for i in range(0, height, x_size):
    for j in range(0, width, y_size):
        s.append(img[i:(i+x_size), j:(j+y_size)])

np.random.shuffle(s)
```

Posteriormente, criamos outra lista (`rows`) onde uniremos os elementos de `s` para formar as linhas da imagem. Fazemos um loop de 0 até `n_rows` e em cada iteração concatenamos `n_cols` elementos de `s` para formar uma posição da nova lista.

Caso a imagem seja dividida em 5 linhas e 10 colunas, por exemplo, a lista `rows` terá 5 posições representando cada uma das linhas. Em cada uma dessas posições concatenamos 10 elementos do vetor `s`.

Por fim, concatenamos os elementos da lista `rows` para formar a nova imagem e a exibimos na tela.

```python
rows = []

for i in range(0, n_rows):
    print(i * n_cols, (i + 1) * n_cols)
    rows.append(np.concatenate(tuple(s[i * n_cols:(i + 1) * n_cols]), 1))

new_img = np.concatenate(tuple(rows[:]))

cv2.imshow("Original Image", img)
cv2.imshow("Shuffle {} x {}".format(n_rows,n_cols), new_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
```

#### 2.3. Resultados

Abaixo podemos ver exemplos da imagem embaralhada com diferentes quantidades de linhas e colunas:

![shuffle1](gabrielsig.github.io/images/shuffle/shuffle1.png)
![shuffle2](gabrielsig.github.io/images/shuffle/shuffle2.png)
![shuffle3](gabrielsig.github.io/images/shuffle/shuffle3.png)









## 3. Contagem de objetos
#### 3.1. Descrição

#### 3.2. Explicando o código

#### 3.3. Resultados

## 4. Equalização de histogramas
#### 4.1. Descrição

#### 4.2. Explicando o código

#### 4.3. Resultados

## 5. Aplicação de filtros
#### 5.1. Descrição

#### 5.2. Explicando o código

#### 5.3. Resultados

## 6. Tilt Shift
#### 6.1. Descrição

#### 6.2. Explicando o código

#### 6.3. Resultados  
