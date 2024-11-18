# TFG-XAI-CONCURRENCIA
El codigo subido genera todas las posibles combinaciones de programas concurrentes, los clasifica y los añade a un archivo csv para su posterior utilización en python, teniendo 3 procesos que realizan funciones distintas:  
- 1º El proceso no influye en la memoria compartida
- 2º El proceso escribe en la memoria compartida
- 3º El proceso lee de la memoria compartida.  
Estos procesos se ha generado según la siguiente gramática:      
![Gramatica datos sinteticos](https://github.com/user-attachments/assets/b4712365-6a8a-45f5-9071-f6a63fc6000b)  
Una vez obtenidos los datos, se han clasificado todas las combinaciones segun la siguiente funcion de clasificación:
![Funcion de clasificacion](https://github.com/user-attachments/assets/527857ec-9827-4a6a-83c4-9cd135ffacf7)  
