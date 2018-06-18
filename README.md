# Magisterka
najlepsza magisterka ever

# Resources
google drive: https://drive.google.com/drive/folders/0B5Qah_avIRO0NUxVT2JlNTF3Rms?usp=sharing

google GPU : https://hackernoon.com/train-your-machine-learning-models-on-googles-gpus-for-free-forever-a41bd309d6ad

# TUTORIALS
http://course.fast.ai/

VIDEOS
-------------
https://www.youtube.com/watch?v=-E2N1kQc8MM - Generating videos ze śmiesznym kolesiem


GITHUBS
-------------
https://github.com/llSourcell/how_to_generate_video - kod śmiesznego kolesia
https://github.com/niazangels/vae-pokedex - generowanie pokemonów
https://github.com/Nemzy/video_generator - generowanie filmików

SUPER RESOLUTION GANs
https://github.com/tadax/srgan
- TensorFlow + VGG19 + Imagenet

GAN Keras
https://github.com/eriklindernoren/Keras-GAN#installation

POTENCJALNE ROZDZIAŁY PRACY MAGISTERSKIEJ
-------------
1. Sieci neuronowe + konwolucyjne sieci neuronowe *- część teoretyczna*
* zbiory danych (opisanie tego, że potrzebna jest dobra jakość + duża ilość obiektów)
* opisanie tego, że algorytmy rozwijają się w stronę potrzebowania coraz mniejszej ilości zdjęć, ponieważ mają zaimplementowane funkcje, które pozwalają je preprocesować
2. Data augumentation *- część teoretyczna*
* preprocesowanie zdjęcia przed zastosowaniem algorytmu sieci neuronowej
* zwiększenie jakości zbioru
* rotowanie zdjęć - zwiększenie ilości zdjęć w zbiorze, ALE kosztem gubienia jakości
3. Super resolution GANs *- część teoretyczna*
* zbudowanie sieci neuronowej pozwalającej na zwiększenie jakości zbioru danych
  * *nie wiem czy budować to od nowa, czy posłużyć się gotową siecią z githuba*
* rotowanie zdjęć i zwiększenie ich jakości sieciami GAN
  * *opis koncepcji*
4. Porównanie efektów *- czyli zastosowanie opisanych powyżej rzeczy w praktyce*
* zbudowanie klasyfikatora na zbiorze danych bez zrotowanych zdjęć
  * ALBO zbudowanie klasyfikatora na zbiorze danych ze zrotowanymi zdjęciami, ale bez zastosowania Super resolution GANs
  * *Klasyfikator chyba najlepiej i najwygodniej zbudować w bibliotece fast.ai *
* zbudowanie klasyfikatora na zbiorze danych ze zrotowanymi zdjęciami i z zastosowaniem super resolution GANs
* porównanie wyników
* wyciągnięcie wniosków

PLAN DZIAŁANIA
-------------
1. Postawić maszynę EC2 p2.xlarge (Deep Learning AMI version 10.0)
2. Połączyć repozytorium z AWSem
3. Postawić środowisko pytorch_mgr
4. Zainstalować na środowisku pytorch_mgr:
* gany
* fast.ai 
5. Pobrać dane (pieski/kotki) i wrzucić je na AWS
6. Skonstruować workflow do trenowania sieci klasyfikujących obrazki
7. Skonstruować workflow do trenowania ganów
8. Skonstruować workflow z kolejnymi etapami (w pętli):
* sieć konstruująca obrazki
* sieć klasyfikująca obrazki
* dodanie wyników do data frame
