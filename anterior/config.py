# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:48:20 2020
@author: eduardo
"""
import os
import time
class Config:
    ENTRENAMIENTO = '\\entrenamiento\\'
    ENTRENAMIENTO_ESTRELLAS = '\\entrenamiento_estrellas\\'
    #RUTA_IMAGENES = ENTRENAMIENTO + 'imagenes_sin_fondo_512_threshold_2000'
    #RUTA_IMAGENES = ENTRENAMIENTO + 'imagenes_sin_fondo_512_threshold_2000'
    RUTA_IMAGENES = ENTRENAMIENTO + 'imagenes_recortadas'
    RUTA_IMAGENES_ESTRELLAS = ENTRENAMIENTO_ESTRELLAS + 'fits_tif_512_pfcm'
    #RUTA_MASCARAS = ENTRENAMIENTO + 'mascaras_512_threshold_2000'
    #RUTA_MASCARAS = ENTRENAMIENTO + 'mascaras_512_threshold_2000'
    RUTA_MASCARAS = ENTRENAMIENTO + 'mascaras_recortadas'
    RUTA_MASCARAS_ESTRELLAS = ENTRENAMIENTO_ESTRELLAS + 'fits_tif_mascaras_512'
    VALIDACION = '\\validar\\'
    VALIDACION_ESTRELLAS = '\\validar_estrellas\\'
    #RUTA_VALIDAR_IMG = VALIDACION + 'imagenes_sin_fondo_512_threshold_2000'
    #RUTA_VALIDAR_IMG = VALIDACION + 'imagenes_sin_fondo_512_threshold_2000'
    RUTA_VALIDAR_IMG = VALIDACION + 'imagenes_recortadas'
    RUTA_VALIDAR_ESTRELLAS_IMG = VALIDACION_ESTRELLAS + 'fits_tif_512_pfcm'
    #RUTA_VALIDAR_MAS = VALIDACION + 'mascaras_512_threshold_2000'
    #RUTA_VALIDAR_MAS = VALIDACION + 'mascaras_512_threshold_2000'
    RUTA_VALIDAR_MAS = VALIDACION + 'mascaras_recortadas'
    RUTA_VALIDAR_ESTRELLAS_MAS = VALIDACION_ESTRELLAS + 'fits_tif_mascaras_512'
    RUTA_MODELO = os.getcwd() + '\\modelo'
    RUTA_MODELO_ESTRELLAS = os.getcwd() + '\\modelo_estrellas'
    RUTA_GRAFICAS = os.getcwd() + '\\graficas'
    RUTA_GRAFICAS_ESTRELLAS = os.getcwd() + '\\graficas_estrellas'
    RUTA_RESULTADOS = os.getcwd() + '\\resultados'
    RUTA_RESULTADOS_ESTRELLAS = os.getcwd() + '\\resultados_estrellas'
    TAMANIO_IMAGENES_X = 512
    TAMANIO_IMAGENES_Y = 512
    __RUTA_AUMENTADAS = os.getcwd() + '\\aumentadas'
    __RUTA_AUMENTADAS_MAS = os.getcwd() + '\\aumentadas_mascaras'
    __RUTA_AUMENTADAS_VAL = os.getcwd() + '\\aumentadas_val'
    __RUTA_AUMENTADAS_VAL_MAS = os.getcwd() + '\\aumentadas_val_mascaras'
    def __init__(self, e, o, ini, f, d, tipo):
        if tipo == 1:
            self.config_directorios(e, o, ini, f, d)
        elif tipo == 2:
            self.config_directorios_estrellas(e, o, ini, d)
    def __crea_directorio__(self, ruta):
        try:
            os.stat(ruta)
        except:
            os.mkdir(ruta)
    def config_directorios_reanudar(self, t, e, o, ini, d):
        self.__tiempoyhora = time.strftime(t)
        #ESTRELLAS
        self.__crea_directorio__(self.RUTA_MODELO_ESTRELLAS)
        self.RUTA_GUARDAR_MODELO_ESTRELLAS = self.RUTA_MODELO_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_MODELO_ESTRELLAS)
        self.__crea_directorio__(self.RUTA_GRAFICAS_ESTRELLAS)
        self.RUTA_GUARDAR_GRAFICAS_ESTRELLAS = self.RUTA_GRAFICAS_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_GRAFICAS_ESTRELLAS)
        self.__crea_directorio__(self.RUTA_RESULTADOS_ESTRELLAS)
        self.RUTA_GUARDAR_RESULTADOS_ESTRELLAS = self.RUTA_RESULTADOS_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_RESULTADOS_ESTRELLAS)
        #AUMENTADAS
        self.__crea_directorio__(self.__RUTA_AUMENTADAS)
        self.RUTA_GUARDAR_AUMENTADAS = self.__RUTA_AUMENTADAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_MAS = self.__RUTA_AUMENTADAS_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_MAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL)
        self.RUTA_GUARDAR_AUMENTADAS_VAL = self.__RUTA_AUMENTADAS_VAL + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS = self.__RUTA_AUMENTADAS_VAL_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS)
    def config_directorios(self, e, o, ini, f, d):
        self.__tiempoyhora = time.strftime("%d_%m_%y_%H_%M_%S")
        #NEBULOSAS
        self.__crea_directorio__(self.RUTA_MODELO)
        self.RUTA_GUARDAR_MODELO = self.RUTA_MODELO + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_MODELO)
        self.__crea_directorio__(self.RUTA_GRAFICAS)
        self.RUTA_GUARDAR_GRAFICAS = self.RUTA_GRAFICAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_GRAFICAS)
        self.__crea_directorio__(self.RUTA_RESULTADOS)
        self.RUTA_GUARDAR_RESULTADOS = self.RUTA_RESULTADOS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_RESULTADOS)
        #AUMENTADAS
        self.__crea_directorio__(self.__RUTA_AUMENTADAS)
        self.RUTA_GUARDAR_AUMENTADAS = self.__RUTA_AUMENTADAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_MAS = self.__RUTA_AUMENTADAS_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_MAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL)
        self.RUTA_GUARDAR_AUMENTADAS_VAL = self.__RUTA_AUMENTADAS_VAL + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS = self.__RUTA_AUMENTADAS_VAL_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS)
    def config_directorios_estrellas(self, e, o, ini, d):
        self.__tiempoyhora = time.strftime("%d_%m_%y_%H_%M_%S")
        #ESTRELLAS
        self.__crea_directorio__(self.RUTA_MODELO_ESTRELLAS)
        self.RUTA_GUARDAR_MODELO_ESTRELLAS = self.RUTA_MODELO_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_MODELO_ESTRELLAS)
        self.__crea_directorio__(self.RUTA_GRAFICAS_ESTRELLAS)
        self.RUTA_GUARDAR_GRAFICAS_ESTRELLAS = self.RUTA_GRAFICAS_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_GRAFICAS_ESTRELLAS)
        self.__crea_directorio__(self.RUTA_RESULTADOS_ESTRELLAS)
        self.RUTA_GUARDAR_RESULTADOS_ESTRELLAS = self.RUTA_RESULTADOS_ESTRELLAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_RESULTADOS_ESTRELLAS)
        #AUMENTADAS
        self.__crea_directorio__(self.__RUTA_AUMENTADAS)
        self.RUTA_GUARDAR_AUMENTADAS = self.__RUTA_AUMENTADAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_MAS = self.__RUTA_AUMENTADAS_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_MAS)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL)
        self.RUTA_GUARDAR_AUMENTADAS_VAL = self.__RUTA_AUMENTADAS_VAL + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL)
        self.__crea_directorio__(self.__RUTA_AUMENTADAS_VAL_MAS)
        self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS = self.__RUTA_AUMENTADAS_VAL_MAS + "\\" + str(e) + "_" + self.__tiempoyhora + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_capas"
        self.__crea_directorio__(self.RUTA_GUARDAR_AUMENTADAS_VAL_MAS)
    def config_directorios_k_fold(self, k, o):
        '''print("*****************************************")
        print("Entrenamiento K-Fold " + str(k) + " " + o)
        print("*****************************************")
        print("Creando carpeta " + self.__tiempoyhora + " en " + self.ENTRENAMIENTO)'''
        #NEBULOSAS
        self.ENTRENAMIENTO_TIEMPO = os.getcwd() + "\\" + self.ENTRENAMIENTO + "\\" + self.__tiempoyhora
        self.__crea_directorio__(self.ENTRENAMIENTO_TIEMPO)
        self.ENTRENAMIENTO_KFOLD = self.ENTRENAMIENTO_TIEMPO + "\\kfold_" + str(k)
        #NEBULOSAS
        #print("Creando carpeta kfold_" + str(k) + " en " + self.ENTRENAMIENTO_TIEMPO)
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD)
        #print("Creando carpeta para imagenes (entrenamiento)")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.ENTRENAMIENTO + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.RUTA_IMAGENES)
        #print("Creando carpeta para las máscaras (entrenamiento)")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.ENTRENAMIENTO + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.RUTA_MASCARAS)
        #print("Creando carpeta para imagenes (validación)")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.VALIDACION + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.RUTA_VALIDAR_IMG)
        #print("Creando carpeta para las máscaras (validación)")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.VALIDACION + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_KFOLD + "\\" + self.RUTA_VALIDAR_MAS)
        self.__crea_directorio__(self.RUTA_GUARDAR_RESULTADOS + "\\kfold_" + str(k) + "_" + o + "\\")
        #print("*****************************************")
    def config_directorios_k_fold_estrellas(self, k, o):
        #ESTRELLAS
        self.ENTRENAMIENTO_ESTRELLAS_TIEMPO = os.getcwd() + "\\" + self.ENTRENAMIENTO_ESTRELLAS + "\\" + self.__tiempoyhora
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_TIEMPO)
        self.ENTRENAMIENTO_ESTRELLAS_KFOLD = self.ENTRENAMIENTO_ESTRELLAS_TIEMPO + "\\kfold_" + str(k)
        #ESTRELLAS
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD)
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.ENTRENAMIENTO_ESTRELLAS + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.RUTA_IMAGENES_ESTRELLAS)
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.ENTRENAMIENTO_ESTRELLAS + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.RUTA_MASCARAS_ESTRELLAS)
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.VALIDACION_ESTRELLAS + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.RUTA_VALIDAR_ESTRELLAS_IMG)
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.VALIDACION_ESTRELLAS + "\\")
        self.__crea_directorio__(self.ENTRENAMIENTO_ESTRELLAS_KFOLD + "\\" + self.RUTA_VALIDAR_ESTRELLAS_MAS)
        self.__crea_directorio__(self.RUTA_GUARDAR_RESULTADOS_ESTRELLAS + "\\kfold_" + str(k) + "_" + o + "\\")