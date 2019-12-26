from tkinter import filedialog as FileDialog #importamos la clase filedialog
from tkinter import * # importamos todas las funciones del módulo tkinter
from io import open # importamos la función open del módulo io para trabajar con ficheros de texto

ruta = "" # para almacenar la ruta del fichero de texto

def nuevo():
	global ruta #variable global hace referencia a la variable externa a la función
	mensaje.set("Nuevo fichero") # mensaje del monitor inferior al pulsar botón
	ruta = ""
	texto.delete(1.0, "end") #borra desde el primer carácter del texto (\n salto linea) hasta el final
	root.title("Editor de texto con tkinter") # Título de la ventana de nuevo archivo

def abrir():
	global ruta
	mensaje.set("Abrir fichero") # mensaje del monitor inferior al pulsar botón
	ruta = FileDialog.askopenfilename(
		initialdir='.', 
		filetype=(("Ficheros de texto", "*.txt"),),
		title="Abrir un fichero de texto")

	if ruta != "": # si la ruta es diferente de nada escrito,es decir, si hay alguna ruta escrita
		fichero = open(ruta, 'r') # Abrimos fichero en modo lectura r
		contenido = fichero.read()
		texto.delete(1.0,'end')
		texto.insert('insert', contenido) # Insertamos el contenido del archivo que abrimos en el widget texto
		fichero.close()
		root.title(ruta + " - Mi editor") #Actualizamos título de la ventana con nueva ruta al abrir fichero teto


def guardar():
	mensaje.set("Guardar fichero") # mensaje del monitor inferior al pulsar botón
	if ruta != "": 
		contenido = texto.get(1.0,'end-1c') # get para recuperar contenido desde carácter 1 hasta el final, menos el último carácter que es un salto de línea
		fichero = open(ruta, 'w+') # Abrimos fichero en modo escritura + lectura
		fichero.write(contenido) # Guardamos el contenido
		fichero.close() # Al cerrar el fichero, se actualiza el texto en el fichero
		mensaje.set("Fichero guardado correctamente") # Mostramos mensaje por pantalla en el monitor inferior
	else:
		guardar_como()

def guardar_como():
	global ruta
	mensaje.set("Guardar fichero como") # mensaje del monitor inferior al pulsar botón
	fichero = FileDialog.asksaveasfile(title="Guardar fichero", mode="w", defaultextension=".txt")
	if fichero is not None: # para saber que hemos abierto el fichero y no hemos apretado cancelar
		ruta = fichero.name # Propiedad de ficheros para saber ruta total del fichero 
		# Guardamos como antes
		contenido = texto.get(1.0,'end-1c')
		fichero = open(ruta, 'w+')
		fichero.write(contenido)
		fichero.close()
		mensaje.set("Fichero guardado correctamente")
	else: 
		mensaje.set("Guardado cancelado")
		ruta = ""
#------------------------------------------#

# Configuración de la raíz
root = Tk()
root.title("Editor de texto con tkinter")


# Menú superior
menubar = Menu(root) # menubar es un widget de tipo Menu
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Nuevo", command=nuevo)
filemenu.add_command(label="Abrir", command=abrir)
filemenu.add_command(label="Guardar", command=guardar)
filemenu.add_command(label="Guardar como", command=guardar_como)
filemenu.add_separator()
filemenu.add_command(label="Salir", command=root.quit)
menubar.add_cascade(menu=filemenu, label="Archivo")

# Cuadro de texto central
texto = Text(root) # texto es un widget de tipo Text (texto largo)
texto.pack(fill="both", expand=1)
texto.config(bd=0, padx=6, pady=4, font=("Consolas",12))

# Monitor inferior
mensaje = StringVar() # mensaje es una variable dinámica
mensaje.set("Bienvenido al Editor de texto con tkinter")
monitor = Label(root, textvar=mensaje, justify='left')  # monitor es un widget de tipo Label
monitor.pack(side="left")


root.config(menu=menubar)

# Bucle de la apliación
root.mainloop()
