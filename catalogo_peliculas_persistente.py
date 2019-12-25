from io import open
import pickle

class Pelicula:
    
    # Constructor de clase
    def __init__(self, titulo, duracion, lanzamiento):
        self.titulo = titulo
        self.duracion = duracion
        self.lanzamiento = lanzamiento
        print('Se ha creado la película:',self.titulo)
        
    # Redefinición del método String. Al imprimir p, devolverá este formato
    def __str__(self):
        return '{} ({})'.format(self.titulo, self.lanzamiento)


class Catalogo:
    
    peliculas = []
    
    # Constructor de clase
    def __init__(self):
        self.cargarFichero() #cargamos fichero binario
        
    def agregar(self,p): 
        for pTemporal in self.peliculas:
            #if pTemporal==p: Si no funciona esta comparación de objetos, debemos hacerlo de otra forma
            if(pTemporal.titulo==p.titulo) and (pTemporal.duracion==p.duacion) and (pTemporal.lanzamiento ==p.lanzamiento):
                return #si existe la película, no la crea, devuelve vacío
        self.peliculas.append(p) #Añadimos una pelicula a la lista de peliculas
        self.guardarFichero() # Guardado automático en el fichero binario
    

    def mostrar(self):
        if len(self.peliculas)==0:
            print("Catálogo vacío.")
            return 
        print("\nCatálogo de películas actualizado")
        for p in self.peliculas:
            print(p)

    def eliminar(self,p):
        for pTemporal in self.peliculas:
            if pTemporal==p: #Si no funciona esta comparación de objetos, debemos hacerlo de otra forma
            #if(pTemporal.titulo==p.titulo) and (pTemporal.duracion==p.duacion) and (pTemporal.lanzamiento ==p.lanzamiento):
                self.peliculas.remove(pTemporal) #Elimina la pelicula indicada p
                self.guardarFichero() # Guardado automático en el fichero binario
                print(p,"eliminada")
                return
            

    #-------------------------------------#
            
    def cargarFichero(self): # Funcón que hace referencia al fichero binario (con pickle)
        fichero=open('cataloguito.pckl.pckl','ab+') # ab+ es append binario con funciones de lectura, no solo escritura.No borra contenido del fichero, y el puntero se encuentra al final. 
        fichero.seek(0)# Posicionamos el puntero en caracter 0, ó principio del fichero
        try: #la primera vez lanzará el except
            self.peliculas= pickle.load(fichero) 
        except:
            #print ("el fichero binario esta vacío")
            pass
        finally:
            fichero.close()
            #del(fichero)
            print("Se han cargado {} peliculas en el fichero binario.".format (len(self.peliculas)))
            
    
    def guardarFichero(self): # Funcón que hace referencia al fichero binario (con pickle)
        fichero= open('catalogo.pckl','wb') #wb escritura binaria. El modo escritura elimina el contenido del fichero al abrirlo, y el puntero se posiciona al principio del fichero, desde donde se empezará a escribir.
        pickle.dump(self.peliculas,fichero) #dump elimina lo que hay en el fichero,y añade lo que indicamos en el paréntesis (fichero)
        fichero.close() # Cuando se cierra el fichero, se actualiza el contenido de este. 
        del(fichero) #Para evitar problemas con Jupyter
    
    """
    NO USAR EL DESTRUCTOR
    #destructor de clase que se ejecuta automáticamente al finalizar el programa. Se eliminan de la memoria todas las instancias de objeto y rutinas de la librería pickle
    def __del__(self):
        self.guardarFichero() # guardado automático
        print("Se ha guardado el fichero")
    """

#Probar código

c= Catalogo()

c.mostrar() 

P1=Pelicula("Cadena Perpetua",142,1995)
P2=Pelicula("El silencio de los corderos",138,1991)
P3=Pelicula("El precio del poder",170,1984)
P4=Pelicula("Psicosis",109,1971)

c.agregar(P1)
c.agregar(P2)
c.agregar(P3)

c=Catalogo()
c.agregar(P4)

c.mostrar() 
c.eliminar(P1)
c.eliminar(P2)
c.eliminar(P3)

c.mostrar() 
c=Catalogo()
