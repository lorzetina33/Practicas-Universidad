# PROYECTO FINAL_ Alpha Frequence_ PDI

# INTEGRANTES:
# 0251717 - Sara Pineda Gómez
# 0257520 - Itzayana Partida Ibarra
# 0259187 - Alejandra Roque Gutiérrez
# 0266043 - Ana Luisa Villeda Anleu
# 0273952 - Lorenzo Zetina Herrera

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

# ________________________________________________________________________________
# ________________________________________________________________________________


#  1. TRANSFORMACIONES DE INTENSIDAD 
 
# Gamma 

def gamma(): 
    gamma_im = cv.imread("images/pukei-pukei.png", 0)

    if gamma_im is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    cv.namedWindow("Transformación Gamma")
    cv.createTrackbar("Valor Gamma", "Transformación Gamma", 1, 12, lambda x: None)      

    while True:
        g = cv.getTrackbarPos("Valor Gamma", "Transformación Gamma")
        g = max(g, 1)

        # Corrección gamma
        normalized = gamma_im / 255.0
        gamma_corrected = np.power(normalized, g)
        uint8_gamma = np.uint8(gamma_corrected * 255)

        # Escalar ambas imágenes a la mitad de su tamaño original
        escala = 0.5  # cambia esto si quieres más o menos pequeño
        original_redimensionada = cv.resize(gamma_im, (0, 0), fx=escala, fy=escala)
        gamma_redimensionada = cv.resize(uint8_gamma, (0, 0), fx=escala, fy=escala)

        # Mostrar juntas
        gamma_display = np.concatenate((original_redimensionada, gamma_redimensionada), axis=1)
        cv.imshow("Transformación Gamma", gamma_display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Logarítmica
def logaritmica():
    log_im = cv.imread("images/odogaron-gang.jpg", 0)

    if log_im is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    ventana = "Transformación Logarítmica"
    cv.namedWindow(ventana)
    cv.createTrackbar("Canal de Color", ventana, 1, 3, lambda x: None)

    while True:
        if cv.getWindowProperty(ventana, cv.WND_PROP_VISIBLE) < 1:
            break

        k = cv.getTrackbarPos("Canal de Color", ventana)

        # Aplicar transformación logarítmica
        log_res = np.log1p(log_im.astype(np.float32))  # log(1 + I)
        log_res = cv.normalize(log_res, None, 0, 255, cv.NORM_MINMAX)
        log_res = log_res.astype(np.uint8)

        # Redimensionar ambas imágenes para que no se vean enormes
        escala = 0.5
        log_im_small = cv.resize(log_im, (0, 0), fx=escala, fy=escala)
        log_res_small = cv.resize(log_res, (0, 0), fx=escala, fy=escala)

        # Mostrar lado a lado
        display = np.hstack((log_im_small, log_res_small))
        cv.imshow(ventana, display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()



# --------------------------------------------------------------

#  2. FILTRADO ESPACIAL 

# filtro pasa bajas (suavizado)

def filtro_pasa_baja():
    try:
        #Llamar imagen
        imagen = cv.imread("images/Tulio_casco.jpg", 0)
        if imagen is None:
            print("No se encontró la imagen en la ruta especificada")
            return  
        
        #Definir kernel 
        kernel_pasaBajas = np.ones((3, 3), np.float32) / 9

        #Aplicar en las imagenes 
        img_norm = cv.filter2D(imagen, -1, kernel_pasaBajas)
        img_gaus = cv.GaussianBlur(imagen, (15, 15), sigmaX=0)

        # Mostrar resultados 
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1), plt.imshow(imagen, cmap='gray')
        plt.title('Original'), plt.axis('off')
        
        plt.subplot(1, 3, 2), plt.imshow(img_norm, cmap='gray')
        plt.title('Suavizado'), plt.axis('off')
        
        plt.subplot(1, 3, 3), plt.imshow(img_gaus, cmap='gray') 
        plt.title('Suavizado - Gaussiano'), plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error en aplicar: {e}")  


# Filtro pasa altas (sharpenning)

def filtro_pasa_alta():
    try:
        #Llamar imagen
        imagen = cv.imread("images/Tulio_casco.jpg", 0)
        if imagen is None:
            print("No se encontró la imagen en la ruta especificada")
            return 

        #Definir kernel
        kernel_laps = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]], dtype=np.float32)
        kernel_realce = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], dtype=np.float32) 

        #Aplicarlo a las imagen 
        img_bordes = cv.filter2D(imagen, -1, kernel_laps)
        img_realce = cv.filter2D(imagen, -1, kernel_realce)

        # Mostrar resultados 
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1), plt.imshow(imagen, cmap='gray')
        plt.title('Original'), plt.axis('off')
        
        plt.subplot(1, 3, 2), plt.imshow(img_bordes, cmap='gray')
        plt.title('Bordes (Laplaciano)'), plt.axis('off')
        
        plt.subplot(1, 3, 3), plt.imshow(img_realce, cmap='gray')
        plt.title('Realce (Sharpening)'), plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error en aplicar: {e}")
    

# --------------------------------------------------------------

#  3. OPERACIONES MORFOLÓGICAS 

# erosión
def erosion():
    imagen = cv.imread("images/letraj.png", 0)
    _, img_binaria = cv.threshold(imagen, 127, 255, cv.THRESH_BINARY)

    cv.namedWindow("Erosión")
    cv.createTrackbar("Tamaño Kernel", "Erosión", 3, 20, lambda x: None)      
    cv.createTrackbar("Iteraciones", "Erosión", 1, 10, lambda x: None)      

    while True:
        k = cv.getTrackbarPos("Tamaño Kernel", "Erosión")
        iterations = cv.getTrackbarPos("Iteraciones", "Erosión")

        k = max(1, k)
        if k % 2 == 0:
            k += 1  

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        erosionada = cv.erode(img_binaria, kernel, iterations=max(1, iterations))

  
        cv.imshow("Erosión", erosionada)

 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# dilatación 
def dilatacion():
    imagen = cv.imread("images/letraj.png", 0)
    if imagen is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    _, img_bin = cv.threshold(imagen, 127, 255, cv.THRESH_BINARY)

    cv.namedWindow("Dilatación")
    cv.createTrackbar("Kernel", "Dilatación", 3, 20, lambda x: None)
    cv.createTrackbar("Iteraciones", "Dilatación", 1, 10, lambda x: None)

    while True:
        k = cv.getTrackbarPos("Kernel", "Dilatación")
        i = cv.getTrackbarPos("Iteraciones", "Dilatación")

        k = max(1, k)
        if k % 2 == 0:
            k += 1
        i = max(1, i)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        dilatada =cv.dilate(img_bin, kernel, iterations=i)

        combinada =cv.hconcat([img_bin, dilatada])

        cv.imshow("Dilatación", combinada)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# apertura y cierre
def apertura_cierre():
    imagen = cv.imread("images/letraj.png", 0)

    _, img_bin = cv.threshold(imagen, 127, 255, cv.THRESH_BINARY)

    ventana = "Apertura y Cierre"
    cv.namedWindow(ventana)
    cv.createTrackbar("Tamaño Kernel", ventana, 3, 20, lambda x: None)
    cv.createTrackbar("Iteraciones", ventana, 1, 10, lambda x: None)

    while True:
        if cv.getWindowProperty(ventana, cv.WND_PROP_VISIBLE) < 1:
            break

        k = cv.getTrackbarPos("Tamaño Kernel", ventana)
        it = cv.getTrackbarPos("Iteraciones", ventana)

        k = max(1, k)
        if k % 2 == 0:
            k += 1
        it = max(1, it)

        kernel = np.ones((k, k), np.uint8)

        apertura = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel, iterations=it)
        cierre = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel, iterations=it)

        combinada = np.hstack((img_bin, apertura, cierre))
        cv.imshow(ventana, combinada)

        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyWindow(ventana)




# -------------------------------------------------------------

# 4. UMBRALIZACIÓN 

def umbralizacion_global():
    img = cv.imread("images/letraj.png", 0)
    if img is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    cv.namedWindow("Umbralización Global")
    cv.createTrackbar("Umbral", "Umbralización Global", 127, 255, lambda x: None)

    while True:
        if cv.getWindowProperty("Umbralización Global", cv.WND_PROP_VISIBLE) < 1:
            break

        umbral = cv.getTrackbarPos("Umbral", "Umbralización Global")
        _, umbralizada = cv.threshold(img, umbral, 255, cv.THRESH_BINARY)

        combinada = cv.hconcat([img, umbralizada])
        cv.imshow("Umbralización Global", combinada)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

def umbral_adaptativo():
    img = cv.imread("images/letraj.png", 0)
    if img is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    cv.namedWindow("Umbralización Adaptativa")
    cv.createTrackbar("Bloque", "Umbralización Adaptativa", 11, 50, lambda x: None)
    cv.createTrackbar("Constante", "Umbralización Adaptativa", 2, 20, lambda x: None)

    while True:
        if cv.getWindowProperty("Umbralización Adaptativa", cv.WND_PROP_VISIBLE) < 1:
            break

        bloque = cv.getTrackbarPos("Bloque", "Umbralización Adaptativa")
        c = cv.getTrackbarPos("Constante", "Umbralización Adaptativa")

        if bloque % 2 == 0:
            bloque += 1  # asegurarse que sea impar
        bloque = max(3, bloque)

        adapt = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, bloque, c)

        combinada = cv.hconcat([img, adapt])
        cv.imshow("Umbralización Adaptativa", combinada)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


# -------------------------------------------------------------

# DETECCIÓN Y ANÁLISIS 

# Componentes conectados 
def componentes_conectados():
    img = cv.imread('images/detect_blob.png')  

    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return

    if len(img.shape) == 2 or img.shape[2] == 1:
        color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        color_img = img.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

    def actualizar(val):
        if cv.getWindowProperty('Componentes', cv.WND_PROP_VISIBLE) < 1:
            return 

        try:
            area_min = cv.getTrackbarPos('Área mínima', 'Componentes')
            altura_min = cv.getTrackbarPos('Altura mínima', 'Componentes')
        except:
            return 

        output = color_img.copy()

        for i in range(1, numLabels):
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]
            cX, cY = int(centroids[i][0]), int(centroids[i][1])

            if area > area_min or h > altura_min:
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(output, (cX, cY), 4, (255, 0, 0), -1)

        cv.imshow('Componentes', output)


    cv.namedWindow('Componentes', cv.WINDOW_NORMAL)
    cv.createTrackbar('Área mínima', 'Componentes', 1300, 10000, actualizar)
    cv.createTrackbar('Altura mínima', 'Componentes', 31, 100, actualizar)

    actualizar(0)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


# Detección de bordes (Canny)
def deteccion_bordes():
    img = cv.imread('images/cards.png')

    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return

    if len(img.shape) == 3:
        src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        src_gray = img.copy()

    ratio = 3
    kernel_size = 3
    max_lowThreshold = 100
    window_name = 'Canny Edge Map'

    def CannyThreshold(val):
        low_threshold = val
        blurred = cv.blur(src_gray, (3, 3))
        edges = cv.Canny(blurred, low_threshold, low_threshold * ratio, kernel_size)
        cv.imshow(window_name, edges)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar('Min Threshold', window_name, 0, max_lowThreshold, CannyThreshold)

    CannyThreshold(0)

    cv.waitKey(0)
    cv.destroyAllWindows()




# ________________________________________________________________________________
# ________________________________________________________________________________

# MENÚ 
def menu():
    while True:
        print("\033[1;34m______________________________________________________________________________________________________________________________")
        print("______________________________________________________________________________________________________________________________")
        print("\n\t\t\t\t\t PROYECTO FINAL - EQUIPO ALPHA FREQUENCE")
        print("______________________________________________________________________________________________________________________________")
        print("______________________________________________________________________________________________________________________________\033[0m")

        print(" \n\tEste programa permite aplicar diversas técnicas de procesamiento de imágenes, utilizando la librería OpenCV. ")
        print(" \tA través de un menú interactivo, podrás elegir entre transformaciones de intensidad, filtrado espacial, opera- ")
        print(" \tciones morfológicas, umbralización, detección de bordes y análisis de componentes conectados. ")
        print(" \n\tCada opción abrirá una ventana donde podrás modificar parámetros en tiempo real usando sliders (trackbars). ")
        print(" \tPresiona la tecla 'q' para cerrar la ventana actual. Al finalizar, se te preguntará si deseas probar otra técnica. ")

        print("\033[1;34m\n\t------ MENÚ PRINCIPAL ------\033[0m")
        print("\n1. Función Gamma ")
        print("2. Transformación Logarítmica")
        print("3. Filtro Pasa Baja")
        print("4. Filtro Pasa Alta")
        print("5. Erosión")
        print("6. Dilatación")
        print("7. Apertura y Cierre")
        print("8. Umbralización Global")
        print("9. Umbralización Adaptativa")
        print("10. Componentes Conectados")
        print("11. Detección de Bordes (Canny)")
        
        
        opcion = input("\033[1;36m\n\tSelecciona una opción (o presiona 'x' para salir): ")
        print("\nPresiona 'q' para cerrar la ventana emergente.")

        if opcion == '1':
            gamma()
        elif opcion == '2':
            logaritmica()
        elif opcion == '3':
            filtro_pasa_baja()
        elif opcion == '4':
            filtro_pasa_alta()
        elif opcion == '5':
            erosion()
        elif opcion == '6':
            dilatacion()
        elif opcion == '7':
            apertura_cierre()
        elif opcion == '8':
            umbralizacion_global()
        elif opcion == '9':
            umbral_adaptativo()
        elif opcion == '10':
            componentes_conectados()
        elif opcion == '11':
            deteccion_bordes()
        elif opcion.lower() == 'x':
            print("\033[31mHas salido del programa...\033[0m")
            break
        else:
            print("\033[31m * Opción inválida * \033[0m ")
            continue  # Volver al menú sin preguntar

        continuar = input("\n¿Deseas probar otra técnica? (s/n): ").strip().lower()
        if continuar != 's':
            print("Gracias por usar el programa.")
            break


if __name__ == "__main__":
    menu()