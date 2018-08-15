# 04.08.2018
Kilka podejść do sprawdzenia w workflow:
1. Jak klasyfikatory zachowują się przy zupełnie różnych od siebie literkach (A oraz D)
2. Jak klasyfikatory zachowują się przy podobnych do siebie literkach (G oraz C)
3. Jak klasyfikatory zachowują się przy bardzo podobnych do siebie literkach (I oraz J)

W każdym z tych przypadków:
1. Sprawdzić klasyfikator przy zbilansowanym zbiorze
	- 1000 obserwacji na jedną literkę, 1000 obserwacji na drugą literkę
	
2. Sprawdzić klasyfikator przy niezbilansowanym zbiorze (najpierw na jedną literkę, później na drugą literkę)
	- 1000 obserwacji na jedną literkę, 500 obserwacji na drugą literkę
2.1. Sprawdzić klasyfikator przy niezbilansowanym zbiorze (najpierw na jedną literkę, później na drugą literkę), ale trenować go zdecydowanie dłużej
2.2. Sprawdzić klasyfikator przy zbiorze zbilansowanym przez GANy (najpierw na jedną literkę, a później na drugą literkę)

3. Sprawdzić klasyfikator przy niezbilansowanym zbiorze (najpierw na jedną literkę, później na drugą literkę)
	- 1000 obserwacji na jedną literkę, 100/200 obserwacji na drugą literkę
3.1. Sprawdzić klasyfikator przy niezbilansowanym zbiorze (najpierw na jedną literkę, później na drugą literkę), ale trenować go zdecydowanie dłużej
3.2. Sprawdzić klasyfikator przy zbiorze zbilansowanym przez GANy (najpierw na jedną literkę, a później na drugą literkę)


#29.07.2018
WHAT WE HAVE
1. FOLDER STRUCTURE
2. LOADING DATA SETS
3. DEFINING DCGAN
4. TRAINING DCGAN
5. MASTER THESIS VERSION 0.1


WHAT WE NEED
1. - 
2. LOADING DATASET PIPELINE
- function for dividing datasets for groups depending on parameters
- changing number of classes (2-3-20)
- changing dataset to mnist ?
3. DEFINING CLASSIFICATORS
- Keras convnet
- 3 others ?
4. PIPELINE
- statistics (GAN + class + final? )
- charts (GAN + class + final? )
5. MASTER THESIS FINAL VERSION
	0. Wprowadzenie?
	1. Metody balansowania próby (d)
		- Tomek
	2. GAN (d)
		- Tomek
	3. Klasyfikatory (d)
		- Karolina
		- Napisac cos o kolejnosci przyjmowania do train generatora obrazkow - czy to ma znaczenie
	4. Opowieść o danych i opowieść o programie (m)
		- ?
		- Pokazać przykłady "prawilnych" literek oraz pokazać to, że zamiast liter pojawiają się np znaczki (rower/zwierzę itd)
	5. Przedstawienie wyników (d)
		- Tomek
	6. Dyskusja (d)
		- Karolina
	7. Bibliografia
	

STREAM 1
- Działamy na dwóch klasach
1. Napisać do Ramszy o pracę o augumentacji - Karolina
2. Dataset - ładowanie danych?? - Tomek
3. Klasyfikator - Tomek
4. Statystyki - Karolina
	- GAN + wykresy

DAILY (max. 10 minut) o 20:30
1. Co zrobiłeś
2. Co Ci przeszkadza
3. Co będziesz robić

-------------



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
* opis problemu: mała ilość danych, problem ze zbiorami, które nie są zbilansowane (np. jest 100 kotów i 10000 psów)
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
