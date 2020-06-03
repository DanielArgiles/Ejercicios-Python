# archivo principal

import mongodb_insert # módulo para insertar datos en la base de datos
import mongodb_read # módulo para leer datos en la base de datos
import mongodb_update # módulo para modificar datos en la base de datos
import mongodb_delete # módulo para eliminar datos en la base de datos



class Programa:
	def __init__(self):
		self.dato=1
	
	def menu(self):

		while self.dato:
			# Menú de selección
			selection= input('\nSelect 1 to insert, 2 to update, 3 to read, 4 to delete\n')
			if selection=='1':
				mongodb_insert.insert() # cargamos el método insert del módulo mongodb_insert
			elif selection=='2':
				mongodb_update.update()
			elif selection=='3':
				mongodb_read.read()
			elif selection=='4':
				mongodb_delete.delete()
			else:
				print('n INVALID SELECTION \n')


persona1=Programa()
persona1.menu()


# Para ejecutar la aplicación necesitamos 2 consolas, una para conectar nuestro servidor, y otra para ejecutar nuestro programa
# En la consola 1: mongod , conectamos nuestro servidor de la base de datos de mongodb 
# En la consola 2: app.py

"""
---------------
APUNTES MONGODB:
MongoDB (humongous:enorme) es un sistema de bases de datos NoSQL, orientado a documentos,desarrollado bajo el concepto de código abierto.

En lugar de guardar los datos en tablas como se hace en las bases de datos relacionales, MongoDB guarda las estructuras de datos en documentos similares a JSON con un esquema dinámico,
(MongoDB utiliza una especificacion llamada RSON), haciendo que la integración de los datos en ciertas aplicaciones sea más fáci y rápida.

Cada registro o conjunto de datos se denomina  documento.
Los documentos se pueden agrupar en colecciones, las cuales se podria decir que son el equivalente a las tablas en una base de datos relacional ( solo que las colecciones pueden almacenar documentos con muy diferentes formatos, en lugar de estar sometidos a un esquema fijo).
Se pueden crear índices para algunos atributos de los documentos, de modo que MongoDB mantendrá una estructura interna eficiente para  el acceso a la información por los contenidos de los atributos.

-Los distintos documentos se almacenan en format BSON (binary JSON), que es una versión modificada de JSON que permite búsquedas rápidas de datos.
Para hacernos una idea, BSON guarda de forma expícita las longitudes de los campos, índices de arrays, y demás información útil para el escaneo de datos.
Es por esto que , en algunos casos, el mismo documento en BSON ocupa un poco más de espacio de lo que ocuparía de estar almacenado directamente en formato JSON.
Pero una de las ideas claves en los sistemas NoSQL, es que el almacenamiento es barato, y es mejor aproveharlo si asi se introduce un considerable incremento en la velocidad de localización de información dentro de un documento.

-Sin embargo, en la práctica, nunca veremos el formato en que verdaderamente se almacenan los datos, y trabajaremos siempre sobre un documento en JSON tanto al almacenar como al consultar información.
Ejemplo documento MongoDB: 

{
    "id":employeeid,
    "name":employeename,
    "age":employeage,
    "country": employeecountry
    
}

-Instalación pymongo_ import pymongo
-Instalación BBDD mongoDB: Conectamos el servidor de nuestra BBDD con la instrucción en consola:mongod
- Los datos de la base de datoss se almacenan en :C:\data\db


-Conexión mongo-client: 
from pymongo import MongoClient
client= MongoClient() # puerto predeterminado
client= MongoClient("localhost",27017)
client=MongoClient('mongodeb://localhost:27017')

-Obtención de una BBDD: 
db=cliente.test_database
db=cliente['prueba de base de datos']

- Conseguir una colección:
coleccion=db.test_collection
de recogida=db['test-coleccion']

- Documentos: 
import datetime
mensaje={"autor":"Mike",
    "texto":"Mi primer blog!",
    "tags":["mongodb","Python","pymongo"],
    "fecha":fecha y hora.fecha y hora.utcnow()}
    
- Inserción de un documento en una colección:
mensajes=db.mensajes
post_id=mensajes.insert_one(poste).inserted_id
post_id
El valor de "id" es único en la colección
insert_one() devuelve una instancia de insertOneResult.
Tras insertar el primer documento, la BBDD mensajes se ha creado en el servidor
Comprobación: db.collection_names(include_system_collections=Falso)

-Conseguir un documento con find_one() (consulta más básica):
import pprint
pprint.pprint (mensajes.find_one())
{u'_id':OBJECTID('...'),
u'author:u'Mike'
u'date':datetime.datetime(...),
u'tags':[u'mongodb',u'python 'u'pymongo'],
u'text'u'My primera entrada del blog!}

# Para limitar los resultados a un documento con el autor "Mike":
pprint.pprint(mensajes.find_one({"autor":"Mike"}))
...
Si tratamos con un autor diferente como "Eliot" no hay resultados

- Consulta por objectid:
post_id
OBJECTD(...)
pprint.pprint(mensajes.find_one({"_id":"post_id"}))
u'author:u'Mike'
u'date':datetime.datetime(...),
u'tags':[u'mongodb',u'python 'u'pymongo'],
u'text'u'My primera entrada del blog!}

post_id_as_str=str(post_id)
mensajes.find_one({"_id":post_id_as_str}) #Ningún resultado


# En aplicacione web,para conseguir una OBJECTID de la URL de solicitud y encontrar el documento correspondiente:
from bson.objectid import OBJECTID
def llegar(post_id):
    documento= cliente.db.coleccion.find_one({"_id":OBJECTID(post_id)})
    
-Antes de empezar con la aplicación, descargamos la BBDD MongoDB en la página web e instalemos pymongo: pip install pymongo
"""