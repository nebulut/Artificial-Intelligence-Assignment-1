import cv2
import numpy as np
import random


#cemberin noktalarini belirleme
def create_circle_points(radius, center):
    points = []
    for angle in range(360):
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        points.append((x, y))
    return points #liste cemberin cevresindeki 360 nokta

#gelen bos cemberin noktalari arasinda dizinin elemanlarina gore cizgi cekme
def draw_lines(image, points, sequence):
    for i in range(len(sequence) - 1):
        start_point = points[sequence[i] - 1]
        end_point = points[sequence[i + 1] - 1]
        cv2.line(image, start_point, end_point, (0, 0, 0), 1)

#duz beyaz arka plani cember ile kaplar
def draw_circle(image, center, radius):
    cv2.circle(image, center, radius, (0, 0, 0), 5)


def fitness_score(image1, image2):
    xor_image = cv2.bitwise_xor(image1, image2) #iki resim pikselleri XOR yapilarak farki buldum
    score = np.sum(xor_image) / 255  #beyaz 255 ile temsil edildigi icin farkli piksel sayisi normalde piksel*255 oluyor
    return score #benzerlik skorunu doner (dusuk = daha iyi benzerlik)

def hill_climbing(target_image, points, max_iterations=100000, K=1000, initial_degradation=100, final_degradation=1):
    current_sequence = [random.randint(1, 360) for _ in range(K)] #rastgele baslangic dizisi
    current_image = np.ones((500, 500), dtype=np.uint8) * 255 #beyaz bos goruntu
    draw_circle(current_image, (250, 250), 250)
    draw_lines(current_image, points, current_sequence)
    _, current_image = cv2.threshold(current_image, 127, 255, cv2.THRESH_BINARY)
    current_score = fitness_score(target_image, current_image)

    for iteration in range(max_iterations):
        #bozulma toleransi lineer bir sekilde gittikce azaliyor
        degradation = initial_degradation - (iteration / max_iterations) * (initial_degradation - final_degradation)

        #tepe tirmanma icin rastgele atlama
        new_sequence = current_sequence.copy()
        random_index = random.randint(0, K-1)
        new_sequence[random_index] = random.randint(1, 360)

        #yeni goruntu olusturuluyor ve fitness skoru hesaplaniyor
        new_image = np.ones((500, 500), dtype=np.uint8) * 255
        draw_circle(new_image, (250, 250), 250)
        draw_lines(new_image, points, new_sequence)
        _, new_image = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)
        new_score = fitness_score(target_image, new_image)

        #skor farki izin verilen bozulmadan kucukse yeni resmi degistiriyoruz
        score_difference = new_score - current_score
        if score_difference < degradation:
            current_sequence = new_sequence
            current_image = new_image
            current_score = new_score

        #her 250 tekrarda bir goruntuleri goster
        if iteration % 250 == 0:
            cv2.destroyAllWindows()
            combined_image = np.hstack((target_image, current_image))
            cv2.imshow(f'Iteration {iteration}', combined_image)
            cv2.waitKey(1)
            print(f'Iteration {iteration}: Similarity Score = {current_score}')  # Benzerlik oranını yazdır

    return current_image, current_sequence


#ana resmi okuma ve binary'e donusturme
target_image = cv2.imread('yildiz.png', cv2.IMREAD_GRAYSCALE)
target_image = cv2.resize(target_image, (500, 500)) #burada resmi 500*500 olacak sekilde standart yapiyorum
_, target_image = cv2.threshold(target_image, 127, 255, cv2.THRESH_BINARY)

#500*500 resme uygun olacak sekilde cemberin merkezi ve yaricapi
center = (250, 250)
radius = 250

#cemberin noktalari belirlenir
points = create_circle_points(radius, center)

#tepe tirmanma ile benzer resmi olusturma
optimized_image, optimized_sequence = hill_climbing(target_image, points)

#sonuclari gosterme
combined_image = np.hstack((target_image, optimized_image))
cv2.imshow('Final Result', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cizilen resmi kaydetme
cv2.imwrite('optimized_image.png', optimized_image)

cv2.destroyAllWindows()
