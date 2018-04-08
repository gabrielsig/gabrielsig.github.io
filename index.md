# Processamento Digital de Imagens

Esta página tem o objetivo de exibir os trabalhos realizados para a disciplina Processamento Digital de Imagens (DCA0445) ministrada pelo professor [Agostinho Brito](http://agostinhobritojr.github.io/).

Todos os códigos foram implementados em Python com a biblioteca OpenCV. Para usa-los basta instalar a biblioteca no seu ambiente Python e realizar a execução normalmente.

Os tutoriais assim como os exercícios disponibilizados pelo professor podem ser vistos [aqui](http://agostinhobritojr.github.io/tutoriais/pdi/).  

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

Nesse programa foi implementado um algoritmo para realizar a contagem de objetos com e sem buracos presentes em uma figura. Para isso usaremos a função `cv2.floodFill()` para marcar e contar tais objetos.

Inicialmente percebemos que a implementação proposta apresentava a limitação de não ser capaz de contar mais do que 255 objetos em uma imagem de 8 bits, pois o valor da contagem é usado para preencher cada objeto encontrado. Uma forma forma de contornar isso seria simplesmente usar um valor fixo para ser usado no preenchimento dos objetos encontrados pela função `cv2.floodFill()`, permitindo a contagem de mais de 255 objetos. Outra forma seria usar uma imagem em ponto flutuante, o que nos gera uma gama infinita de valores possíveis e, portanto, também eliminaria o problema.

O código usado para a resolução desse problema pode ser encontrado [aqui]()

#### 3.2. Explicando o código

Começamos removendo os objetos que tocam a borda da imagem. Para isso usamos loops aninhados para percorrer todos os pixels das bordas da imagem, aplicando a função `cv2.floodFill()` em todos que possuam intensidade 255. Ao aplicarmos a função usamos a intensidade 0 para preencher cada um dos objetos, de forma que eles passam a ser incorporados ao background a imagem. Além disso, para cada objeto encontrado, incrementamos um contador que será usado para imprimir na tela a quantidade de objetos removidos.  

```python
# load the image
img = cv2.imread('media/bolhas.png', 0)

# show original image
cv2.imshow('Original', img)

# remove objects touching the border of the image
n_removed_objects = remove_border_objects(img)
print("Number of objects removed: ", n_removed_objects)
```

```python
def remove_border_objects(image):
    height, width = image.shape[:2]
    n_removed_objects = 0
    seed = [0, 0]

    for x in [0, height-1]:
        for y in range(0, width):
            if image[x, y] == 255:
                n_removed_objects += 1
                seed = [y, x]
                cv2.floodFill(image, None, tuple(seed), 0)

    for y in [0, width-1]:
        for x in range(0, height):
            if image[x, y] == 255:
                n_removed_objects += 1
                seed = [y, x]
                cv2.floodFill(image, None, tuple(seed), 0)

    # create a copy of the image to write the text and show
    image_copy = image.copy()

    # Write how many objets were removed from the image in the copy of the image
    cv2.putText(image_copy, str(n_removed_objects) + ' Objects removed',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Border removed', image_copy)

    return n_removed_objects
```

Agora devemos contar quantos objetos restam na imagem. Usamos novamente loops aninhados, mas dessa vez percorremos todos os pixels da imagem, aplicando novamente a função `cv2.floodFill()` para marcar os objetos encontrados com intensidade 255. Porém, dessa vez, usamos o valor 100 para preencher os objetos encontrados, dessa forma eles não serão contados mais de uma vez e nem possuirão o mesmo tom do background. Assim como quando removemos os objetos das bordas, incrementamos um contador a cada objeto encontrado para que possamos imprimir a contagem final na tela após o término dos loops.

```python

# label and count the total number of objects
n_objects = count_objects(img)
print("Number of objects: ", n_objects)

```

```python

def count_objects(image):
    height, width = image.shape[:2]
    seed = [0, 0]
    n_objects = 0

    for x in range(0, height):
        for y in range(0, width):
            if img[x, y] == 255:
                n_objects += 1
                seed = [y, x]
                cv2.floodFill(img, None, tuple(seed), 100)

    # create a copy of the image
    image_copy = image.copy()

    # Write the object count
    cv2.putText(image_copy, str(n_objects) + ' Objects',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Object Count', image_copy)

    return n_objects

```

E, por fim, contamos os objetos que possuem furos, mas para isso precisamos realizar alguma etapas extras. Começamos por usar a função `cv2.floodFill()` para, a partir da borda superior da imagem, preencher completamente o background com o valor 255. Dessa forma, os únicos pixels restantes com um valor 0 são aqueles localizados no interior de um objeto.

Usamos dois loops aninhados para percorrer novamente toda a imagem. Ao encontrar um pixel com valor 0, significa que encontramos um buraco, mas precisamos verificar se ele pertence a um objeto que já contamos ou não. Para isso verificamos o pixel imediatamente anterior ao encontrado: caso ele tenha um valor diferente do background (atualmente 255), significa que o buraco encontrado está no interior de um objeto não contado, então nós aplicamos o `cv2.floodFill()` para mudar o valor tanto dos pixels do buraco quanto do objeto para 255 e incrementamos o contador. Agora, caso o pixel vizinho ao buraco tenha o valor do background, significa que o objeto no qual ele estava inserido já foi contado e possuía múltiplos furos. Dessa forma, aplicamos o `cv2.floodFill()` para mudarmos o valor para 255, mas não incrementamos o contador de objetos.

Após o término dos loops, nós teremos uma imagem com o fundo branco e apenas os objetos sem furos com valor 100. Aplicamos novamente o `cv2.floodFill()` na borda superior da imagem para mudarmos novamente o valor do background para 0 e subtraímos essa imagem da imagem com todos os objetos. Dessa forma, terminamos com uma imagem que possui apenas os objetos com furos.

```python

# count the number of objects with holes
n_holes = count_holes(img)
print("Number of objects with holes: ", n_holes)

cv2.waitKey(0)

```

```python

def count_holes(image):
    height, width = image.shape[:2]
    seed = [0, 0]
    n_holes = 0

    # create a copy of the image
    image_copy = image.copy()

    # floodfill the background of the image with 255, the the only pixels with value equal to 0 are inside an object
    cv2.floodFill(img, None, (0, 0), 255)

    for x in range(0, height):
        for y in range(0, width):
            # if we find a pixel with 0, we check if its neighbour is an object or the new 255 background
            if img[x, y] == 0:
                # if its the background, it means that we have already counted the object and that it had multiple holes
                if img[x-1, y] == 255:
                    seed = [y, x]
                    cv2.floodFill(img, None, tuple(seed), 255)
                # if its different then the beackground, it means that we have found a new object with a hole in it
                else:
                    n_holes += 1
                    seed = [y, x]
                    cv2.floodFill(img, None, tuple(seed), 255)
                    # we also fill the object
                    seed = [y, x-1]
                    cv2.floodFill(img, None, tuple(seed), 255)

    # floofill the background back to zero
    cv2.floodFill(img, None, (0, 0), 0)

    # subtract the image with only the objects without holes from the original copy previously created
    image_copy = image_copy - image

    # write the number of objects with holes
    cv2.putText(image_copy, str(n_holes) + ' Objects with holes',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Objects with holes', image_copy)

    return n_holes

```

#### 3.3. Resultados

Abaixo podemos ver as imagens geradas pelo algoritmo nas diversas etapas da execução:

![labeling1](gabrielsig.github.io/images/labeling/labeling1.png)

## 4. Equalização de histogramas
#### 4.1. Descrição

Esse exercício tem como objetivo o desenvolvimento de um programa que realize a equalização de imagens antes de exibi-las na tela. Foram desenvolvidas duas variações: uma para fotos e outra para vídeos capturados pela webcam.

Quando trabalhamos com fotos, foi possível desenvolver uma função para gerar o histograma acumulado da imagem e posteriormente usá-lo como função de transformação para gerar a imagem equalizada. Porém, ao trabalharmos com vídeo da webcam, houveram problemas de performance ao tentar usar esse método. Portanto, o mesmo foi substituído pela função do openCV `cv2.equalizeHIst()`, por ser mais otimizada.

Além disso, os dois códigos possuem poucas diferenças, e estas serão discutidas mais a frente quando os os mesmos forem explicados.

O algoritmo para imagens pode ser encontrado [aqui]() e para videos [aqui]().

#### 4.2. Explicando o código

Primeiro discutiremos o algoritmo para fotos, já que a equalização feita nele foi alcançada por meio de funções desenvolvidas a partir da teoria vista em sala de aula.

Inicialmente carregamos a imagem em tons de ciza e calculamos seu histograma por meio da função `cv2.calcHist()`. Então passamos esse histograma como argumento para a função `calc_accumulated_hist()` a fim de obtermos o histograma acumulado.

Essa função cria uma nova lista e usa um loop para concatenar novos elementos na forma de uma soma acumulada, onde cada elemento é a soma de todos os anteriores.

```python

img = cv2.imread('media/farol.jpg', 0)

height, width = img.shape[:2]

# number of bins of the histogram
nbins = 256

# calculate the histogram
hist = cv2.calcHist([img], [0], None, [nbins], [0, 256])

# calculate the accumulated histogram
accum_hist = calc_accumulated_histogram(hist)

```

```python

def calc_accumulated_histogram(histogram):
    accum_hist = []
    accum_hist.append(int(histogram[0]))
    for i in range(1, len(histogram)):
        accum_hist.append(accum_hist[i - 1] + int(histogram[i]))

    return accum_hist

```

Passamos o histograma accumulado e a referência da imagem para a função `equalize()`, que tem como retorno a imagem equalizada.

Essa função usa cada elemento do histograma acumulado para realizar a transformação da imagem. Esse processo gera uma nova imagem com tons mais espalhados pelo espectro, ou seja: o histograma da imagem equalizada tende a ser mais normalizado e, portanto, o histograma acumulado se aproxima de linear.

```python
# equalize the image
eq_img = equalize(accum_hist, img)

```

```python
def equalize(accum_hist, img):

    eq_img = img.copy()

    for x in range(0, height):
        for y in range(0, width):
            # the index in the histogram is the intensity value at (x, y)
            index = img[x, y]
            eq_img[x, y] = np.round(accum_hist[index] * 255 / (height * width))

    return eq_img

```
Calculamos o histograma da imagem equalizada tal como seu histograma acumulado, assim podemos plotar gráficos para comparar a imagem antes e após o processo. Além disso fazemos a normalização dos dois histogramas acumulados para facilitar a visualização dos gráficos.

Por fim mostramos as figuras e os gráficos.

```python
# calculate the histogram of the new image
hist2 = cv2.calcHist([eq_img], [0], None, [nbins], [0, 256])

# calculate the accumulated histogram of the equalized image
accum_hist2 = calc_accumulated_histogram(hist2)

# normalize the accumulated histograms to show the graphs
accum_hist_norm = [x / (height*width) for x in accum_hist]
accum_hist_norm2 = [x / (height*width) for x in accum_hist2]


cv2.imshow('original', img)
cv2.imshow('equalized', eq_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.title('Accumulated histograms (normalized)')
plt.plot(accum_hist_norm, 'k')
plt.plot(accum_hist_norm2, 'b')
plt.xlim([0, nbins])
plt.show()

plt.title('Histograms')
plt.plot(hist, 'k')
plt.plot(hist2, 'b')
plt.xlim([0, nbins])
plt.show()
```

O código para vídeos é bastante similar, só que ao invés de usarmos a função `equalize()` usamos `cv2.equalizeHist()`, e usamos `cap = cv2.VideoCapture()` para abrir a câmera em conjunto com `ret, frame = cap.read()` dentro de um loop para capturar os frames em tempo real. O código completo pode ser visto abaixo:

```python

cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('height = {}, width = {}'.format(height, width))

nbins = 256

while True:
    ret, frame = cap.read()

    # convert the frame to black and white and then to floating point
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate the histogram
    hist = cv2.calcHist([frame_gray], [0], None, [nbins], [0, 256])

    # calculate the accumulated histogram
    accum_hist = []
    accum_hist.append(int(hist[0]))
    for i in range(1, len(hist)):
        accum_hist.append(accum_hist[i - 1] + int(hist[i]))

    # equalize the image
    eq_frame_gray = cv2.equalizeHist(frame_gray)

    cv2.imshow('Original', frame_gray)
    cv2.imshow('Equalized', eq_frame_gray)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close without saving
    if key == 27:
        break
    # if the space bar is pressed, save and close
    # elif key == 32:
    #     break

cap.release()
cv2.destroyAllWindows()
```

#### 4.3. Resultados

Logo abaixo podemos ver a imagem original em conjunto com a imagem obtida após o processo de equalização. O aumento do contraste na segunda imagem é visível, o que denota um espalhamento dos tons no histograma:

![hist1](gabrielsig.github.io/images/histogram/hist1.png)

Analisando os gráficos dos histogramas e histogramas acumulados, podemos concluir que esse é mesmo o caso: podemos ver que o histograma da imagem equalizada (em azul) está mais uniformemente distribuído do que o da imagem antes do processo (em preto). Além disso, vemos que o histograma acumulado tende a uma reta, assim como sugerido anteriormente.

![hist_graph](gabrielsig.github.io/images/histogram/hist_graph.png)

![accum_hist_graph](gabrielsig.github.io/images/histogram/accum_hist_graph.png)

Por fim, vemos o resultado do mesmo processo aplicado a um vídeo da webcam:

![hist2](gabrielsig.github.io/images/histogram/hist2.png)

## 5. Detector de movimento
#### 5.1. Descrição

Esse programa tem como objetivo detectar movimentos em uma cena capturada pela webcam. Para isso calculamos a componente vermelha do histograma da imagem e o comparamos com um histograma anterior para detectar diferenças significativas.

Existem vários métodos para comparar histogramas, optamos por usar o método chi-square por ter se mostrado mais estável nas condições de teste estabelecidas. Além disso, foram implementadas duas trackbars para que o usuário possa mudar dinamicamente o valor de threshold e a quantidade de frames que o programa deve esperar até mudar o histograma usado na comparação da cana.

O código completo do detector de movimento pode ser encontrado [aqui]()

#### 5.2. Explicando o código

Inicialmente abrimos a câmera, capturamos o primeiro frame, aplicamos um filtro de média para reduzir um pouco o ruído da câmera e então calculamos e normalizamos o histograma da componente vermelha. Esse histograma será usado como primeiro histograma de comparação.

```python
cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

nbins = 256

# capture the first frame
ret, frame = cap.read()

# slightly blur the frame to reduce noise from the camera
frame = cv2.blur(frame, (3, 3))

# calculate the red component histogram and normalize it
old_r_hist = cv2.calcHist([frame], [2], None, [nbins], [0, 256])
cv2.normalize(old_r_hist, old_r_hist, cv2.NORM_MINMAX).flatten()
```

Criamos as duas trackbars e as variáveis que serão ajustadas por elas. A primeira trackbar será usada para ajustar o threshold e a segunda para ajustar o delay (em número de frames) até que o programa atribua um novo valor para  ´old_r_hist´. Dessa forma esperamos que o programa possa ser adaptado para diferentes condições de iluminação do ambiente mais facilmente.

```python
iteration = 0
threshold = 5
frames_delay = 10

# create the window an the trackbars
cv2.namedWindow('motion detection')
cv2.createTrackbar('Threshold adjuster', 'motion detection', 5, 50, adjust_threshold)
cv2.createTrackbar('Histogram calculation delay\n(in number of frames)', 'motion detection', 10, 50, hist_calc_delay)
```

```python
def adjust_threshold(slider_pos):
    global threshold
    threshold = slider_pos


def hist_calc_delay(slider_pos):
    global frames_delay
    frames_delay = slider_pos
```

Então entramos em um loop onde um repetimos o processo descrito anteriormente para cada novo frame. O histograma desse novo frame é comparado pelo método de chi-square com o último histograma armazenado. Se o resultado da comparação for maior do que o threshold estabelecido pelo usuário, o texto `MOTION DETECTED!` é impresso na tela.

Por fim, verificamos se a quantidade estabelecida de frames já foi excedida. Em caso positivo, copiamos o histograma atual para a variável `old_r_hist` e ele passa a ser usado nas comparações até que o limite seja excedido novamente.


```python
while True:
    ret, frame = cap.read()

    iteration += 1

    # slightly blur the frame to reduce noise from the camera
    frame = cv2.blur(frame, (3, 3))

    # calculate the red component of the new histogram
    new_r_hist = cv2.calcHist([frame], [2], None, [nbins], [0, 256])
    cv2.normalize(new_r_hist, new_r_hist, cv2.NORM_MINMAX).flatten()

    # compare the histograms with chi-squared method
    result = cv2.compareHist(old_r_hist, new_r_hist, cv2.HISTCMP_CHISQR)

    # if the result is larger then the threshold, some movement was detected
    if result >= threshold:
        cv2.putText(frame, 'MOTION DETECTED!',
                    org=(10, height-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)

    # change the old histogram after the set amount of frames has passed
    if iteration >= frames_delay:
        # put the current histogram as the old histogram for the next comparison
        old_r_hist = new_r_hist.copy()
        iteration = 0

    # show the current frame
    cv2.imshow('motion detection', frame)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close the program
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

#### 5.3. Resultados

Ajustamos os parâmetros até chegarmos em uma configuração boa para a cena em questão. Abaixo podemos ver o output do programa quando não há movimentos na cena e logo em seguida quando um movimento foi detectado:

![motion_detector1](gabrielsig.github.io/images/motion/motion_detector1.png)

![motion_detector2](gabrielsig.github.io/images/motion/motion_detector2.png)

## 6. Aplicação de filtros espaciais
#### 6.1. Descrição

Esse programa tem como objetivo a aplicação de filtros em vídeos capturados pela webcam, mais precisamente a implementação do filtro laplaciano do gaussiano.

Assim como no programa anterior, usaremos trackbars para garantir ao usuário mais controle sobre o programa: A primeira possibilita a escolha de um entre os 6 filtros diferentes; A segunda fornece a possibilidade de visualizar o resultado da filtragem de forma normalizada ou absoluta; E a última possibilita a escolha da intensidade do filtro, ou quantas vezes a máscara será convolucional com a imagem (mais útil para os filtros de borramento).

O código completo do programa de filtragem espacial pode ser encontrado [aqui]()

#### 6.2. Explicando o código

Começamos inicializando a câmera como de costume, instanciando a janela e as trackbars, tal como declarando as variáveis que serão modificadas por elas.

```python
cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# default filter is no filter
filter_type = 0
absolute = 0
filter_intensity = 1
menu_slider_name = '0: No filter\n' \
                   '1: Mean\n' \
                   '2: Gaussian\n' \
                   '3: Vertical Sobel\n' \
                   '4: Horizontal Sobel\n' \
                   '5: Laplacian\n' \
                   '6: Laplacian of Gaussian'
absolute_slider_name = '0: Normalized\n' \
                       '1: Absolute'

cv2.namedWindow('spatial filter')
cv2.createTrackbar(menu_slider_name, 'spatial filter', 0, 6, on_menu_change)
cv2.createTrackbar(absolute_slider_name, 'spatial filter', 0, 1, on_absolute_change)
cv2.createTrackbar('Filter intensity (+1)', 'spatial filter', 0, 9, on_intensity_change)
```

```python
def on_menu_change(slider_pos):
    global filter_type
    filter_type = slider_pos


def on_absolute_change(slider_pos):
    global absolute
    absolute = slider_pos


def on_intensity_change(slider_pos):
    global filter_intensity
    filter_intensity = slider_pos + 1
```
Com isso entramos no loop onde a lógica do programa é executada. Primeiro carregamos o frame atual da câmera e o convertemos para tons de cinza e `float32`.  Então usamos a função `aaply_filter` para aplicar o filtro selecionado pelo usuário nessa imagem em ponto flutuante (o funcionamento detalhado dessa função será descrita mais à frente).

Após obtermos o resultado do filtro, verificamos se o usuários selecionou a opção `absolute`. Em caso positivo, usamos a função `np.absolute()` para obtermos a imagem com valores absolutos. Já em caso negativo, usamos a função `cv2.normalize()` para normalizar a imagem.

Por ultimo, convertemos a imagem de volta para `uint8` e a exibimos na tela.

```python
while True:
    ret, frame = cap.read()

    # convert the frame to black and white and then to floating point
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray32 = np.array(frame_gray, dtype=np.float32)

    # apply the filter
    filter_output = apply_filter(frame_gray32, filter_type, filter_intensity)

    # check if the absolute option is True
    if absolute:
        filter_output = np.absolute(filter_output)
    else:
        filter_output = cv2.normalize(filter_output, None, 0, 255, cv2.NORM_MINMAX)

    # convert the result back to a uint8 image
    result = np.array(filter_output, dtype=np.uint8)

    # show processed image
    cv2.imshow('spatial filter', result)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()
```
A função `apply_filter()` recebe como parâmetros a imagem em ponto flutuante, o tipo de filtro (inteiro de 0 a 6 definido pela trackbar) e o número de passadas da máscara (inteiro de 1 a 10 também definido pela trackbar).

Criamos uma máscara 3x3 de zeros que, de acordo com o filtro selecionado pelo usuário, recebe os valores apropriados para cada caso. No caso do filtro laplaciano do gaussiano, a mascara inicialmente assume os valores da mascara do filtro gaussiano.

Com a mascara definida, usamos a função `cv2.filter2D()` para realizar as convoluções com a imagem. Caso o número de passadas selecionada pelo usuário for maior do que 1, entramos em um loop para realizar a aplicação do filtro múltiplas vezes.

Por último, verificamos se o usuário selecionou a opção 6, que se refere ao filtro laplaciano do gaussiano. Nesse caso, substituímos a máscara pela laplaciana e  usamos a função `cv2.filter2D()` para aplicamos novamente o filtro a imagem com essa nova máscara. Assim somos capazes de obter o resultado da aplicação composta do filtro gaussiano e laplaciano.

```python
def apply_filter(src_img, filter_type, n_passes=1):

    # create a generic 3x3 mask
    mask = np.zeros([3, 3])

    # check which filter was selected and apply the correponding mask
    if filter_type == 0:
        # no Filter
        return src_img
    elif filter_type == 1:
        # mean filter
        # cv2.blur()
        mask = np.ones([3, 3]) / (3**2)
    elif filter_type == 2:
        # gaussian filter
        # cv2.GaussianBlur()
        # cv2.getGaussianKernel()
        mask = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16
    elif filter_type == 3:
        # vertical sobel border detector
        # cv2.Sobel()
        mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    elif filter_type == 4:
        # horizontal sobel border detector
        # cv2.Sobel()
        mask = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    elif filter_type == 5:
        # laplacian filter
        # cv2.Laplacian()
        mask = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
    elif filter_type == 6:
        # laplacian of gaussian filter, so we first use the gaussian mask
        mask = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16
    else:
        print('[ERROR]: could not select one of the available filters')

    # apply the filter with the selected mask
    out_img = cv2.filter2D(src_img, -1, mask)

    # apply multiple times if n_passes > 1
    if n_passes > 1:
        for n in range(0, n_passes-1):
            out_img = cv2.filter2D(out_img, -1, mask)

    # if we select the laplacian of gaussian, we replace the gaussian mask with the laplacian
    # and apply the filter again
    if filter_type == 6:
        mask = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
        out_img = cv2.filter2D(out_img, -1, mask)

    return out_img
```
#### 6.3. Resultados

Abaixo podemos ver a aplicação do filtro laplaciano e logo em seguida do filtro laplaciano do gaussiano, ambos exibidos com a opção `absolute` selecionada.

Podemos ver claramente a presença de muito ruído quando é aplicado apenas o filtro laplaciano. Esse não o caso da segunda imagem, pois o ruído é suavizado pelo borramento realizado pelo filtro gaussiano quando aplicamos o laplaciano do gaussiano.

![filter1](gabrielsig.github.io/images/filter/filter1.png)

![filter1](gabrielsig.github.io/images/filter/filter2.png)

## 7. Tilt Shift
#### 7.1. Descrição

Esse exercício tem como objetivo a aplicação de um efeito tilt shift em uma imagem, que faz com que os objetos tenham uma aparência de miniatura. Esse efeito é obtido aplicando um filtro de borramento nas bordas da imagem e pode ser melhorado se elevarmos a saturação da mesma.

Assim como nas questões anteriores, usaremos trackbars para fornecer um controle mais dinâmico ao usuário: a primeira é responsável por ajustar a linha de foco da imagem; a segunda para regular o tamanho da janela de foco; a terceira regula o decaimento do borramento; a quarta a intensidade desse borramento; e por último, a quinta nos dá a possibilidade de elevar a saturação da imagem.

O código completo usado nesse exercício pode ser obtido [aqui]()

#### 7.2. Explicando o código

Começamos carregando a imagem e fazendo duas cópias: uma delas borrada e outra exatamente igual, que servirá para armazenar o resultado dos processamentos. Depois criamos duas máscaras preenchidas com zeros e do mesmo tamanho da imagem.

Então inicializamos as variáveis que determinam o centro do foco, a abertura da janela e o decaimento do borramento.

Instanciamos a janela e as trackbars e fazemos uma chamada a função `update_tiltshift()` para que a imagem exibida na tela seja atualizada (o funcionamento dessa função será discutido posteriormente)

Por fim, entramos em um loop cuja única função é exibir na tela a imagem atualizada e a máscara usada para obtê-la. Caso o usuário pressione `esc`, o programa é fechado sem salvar a imagem, e caso o usuário pressione a `barra de espaço` o programa salva a imagem processada e fecha.

```python
# load the image, create a copy, a blurred copy and another copy to store the final result
img = cv2.imread('media/rome3.jpg')
img_copy = img.copy()
tiltshifted_img = img.copy()
blurred_img = cv2.blur(img, (5, 5))

img_height, img_width = img.shape[:2]

# create the mask and the negative mask with the same size as the image
mask = np.zeros(img.shape, dtype=np.float32)
mask_negative = mask.copy()

center_line = int(img_height / 2)
l1 = center_line - int(img_height / 4)
l2 = center_line + int(img_height / 4)
decaimento = 50

cv2.namedWindow('Tiltshift')
cv2.createTrackbar('Focus center line', 'Tiltshift', 50, 100, on_focus_change)
cv2.createTrackbar('Focus window height', 'Tiltshift', 50, 100, on_window_height_change)
cv2.createTrackbar('Blur gradient', 'Tiltshift', 50, 100, on_gradient_change)
cv2.createTrackbar('Blur intensity', 'Tiltshift', 0, 10, on_blur_change)
cv2.createTrackbar('Saturation', 'Tiltshift', 0, 5, on_saturation_change)

update_tiltshift()

while True:
    # display the image
    cv2.imshow('Tiltshift', tiltshifted_img)
    cv2.imshow('mask', mask)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close without saving
    if key == 27:
        break
    # if the space bar is pressed, save and close
    elif key == 32:
        cv2.imwrite('media/Tilt_shifted.jpg', tiltshifted_img)
        break

# destroy all windows
cv2.destroyAllWindows()
```
A função `update_tiltshif()` é chamada sempre que alguma das trackbars sofre qualquer alteração. Ela é responsável por re aplicar a função `apha()` em todas as linhas da máscara e fazer uma adição ponderada na forma `output_img = input_img * mask + blurred_img * (1 - mask)`, ou seja: a imagem é multiplicada pela máscara e somada com a imagem borrada multiplicada pelo negativo da máscara. Isso nos dá o efeito tilt shift desejado.

```python
# x = linha da imagem onde o alpha deve ser calculado
# l1,l2 = linhas onde o valor de alpha eh 0.5
# d = forca do decaimento
def alpha(x, l1, l2, d):
    if d == 0:
        d += 0.0001
    return (1/2)*(np.tanh((x-l1)/d) - np.tanh((x-l2)/d))


def update_tiltshift():
    global mask, tiltshifted_img

    # update the mask and the processed image
    for x in range(0, img_height):
        alpha_x = alpha(x, l1, l2, decaimento)

        mask[x, :] = alpha_x
        tiltshifted_img[x, :] = cv2.addWeighted(img_copy[x, :], alpha_x, blurred_img[x, :], 1-alpha_x, 0)
```

As funções chamadas pelas trackbars podem ser vistas a seguir. Cada uma delas é responsável por alterar alguma das variáveis relacionadas ao efeito de acordo com a posição do indicador.

```python
def on_saturation_change(slider_pos):
    global img_copy
    # saturation multiplier ranging from 1.0 to 3.5 in increments of 0.5
    saturation_multiplier = (slider_pos * 2.5 / 5)+1
    # change the color space to HSV so we can easly modify the saturation
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    # split the planes
    (h, s, v) = cv2.split(img_copy)
    # multiply the saturation with the selected multiplier
    s = s * saturation_multiplier
    # nomalize values that fall outside the 0 to 255 range
    s = np.clip(s, 0, 255)
    # merge the planes again and convert the image back to BGR color space
    img_copy = cv2.merge([h, s, v])
    img_copy = cv2.cvtColor(img_copy.astype("uint8"), cv2.COLOR_HSV2BGR)
    # update the blurred image and the tilt shift
    blur_intensity_slider = cv2.getTrackbarPos('Blur intensity', 'Tiltshift')
    on_blur_change(blur_intensity_slider)
    update_tiltshift()


def on_focus_change(slider_pos):
    global center_line
    center_line = int(slider_pos * (img_height - 1) / 100)

    # update the window height with the new position of the focus center
    focus_height_slider = cv2.getTrackbarPos('Focus window height', 'Tiltshift')
    on_window_height_change(focus_height_slider)

    # update the image
    update_tiltshift()


def on_window_height_change(slider_pos):
    global l1, l2

    window_height = int(slider_pos * (img_height - 1) / 100)
    l1 = center_line - int(window_height/2)
    l2 = center_line + int(window_height/2)
    update_tiltshift()


def on_gradient_change(slider_pos):
    global decaimento
    decaimento = slider_pos
    update_tiltshift()


def on_blur_change(slider_pos):
    global blurred_img
    blurred_img = cv2.blur(img_copy, (5, 5))
    for i in range(0, slider_pos):
        blurred_img = cv2.blur(blurred_img, (5, 5))
    update_tiltshift()
```

#### 7.3. Resultados  

Abaixo podemos ver a imagem original, seguida da máscara aplicada no tiltshift e a imagem após o processamento

![tiltshift1](gabrielsig.github.io/images/tiltshift/tiltshift1.jpg)

![mask1](gabrielsig.github.io/images/tiltshift/mask1.png)

![tiltshift2](gabrielsig.github.io/images/tiltshift/tiltshift2.png)
