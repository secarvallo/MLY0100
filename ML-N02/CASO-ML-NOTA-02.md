Intrucciones:  
Eres un experto en Data Science and Data Engieniere con conocimientos extensos en machine learning y algoritmos de aprendizaje automatico, te has desarrollado con larga trayectoria en el rubro de las app de Drivers como uber, Didi , Etc. necesito que tu amplio conocimiento lo apliques en este proyecto. Utilizaremos metodologia CRISP-DM, con herramientas como Google collab y lenguajes Python para la estadistica y codigos de Ml.

Caso de estudio: Optimización del transporte en Nueva York:

La Comisión de Taxis y Limusinas de Nueva York (TLC) dispone de registros masivos de viajes de taxis amarillos, taxis verdes, vehículos de alquiler (FHV) y vehículos de alto volumen (HVFHV). Se busca analizar estos datos para mejorar la eficiencia del transporte, la experiencia del pasajero y la planificación urbana.

El análisis de estos datasets permite estudiar patrones de movilidad, optimizar recursos, anticipar la demanda y detectar comportamientos anómalos, ofreciendo un enfoque integral de gestión de transporte urbano.

Problemáticas que se pueden abordar con los datasets

## Nota 02

#### Problema

**1.**    **Estimación de la duración y tarifa del viaje
(Supervisado – Regresión)**

Problema:

Las tarifas y duraciones de los viajes pueden variar significativamente según la hora del día, la distancia y la zona. Los pasajeros buscan información precisa para planificar sus traslados.

Objetivo:

Predecir la tarifa total o la duración estimada de un viaje.

Mejorar la transparencia y confianza del pasajero.

Ayudar a la TLC a optimizar rutas y
reducir tiempos de espera.

Dataset
aplicable: Yellow Taxi, Green Taxi

#### Dataset

**1. Green Taxi Trip Records (LPEP)**

Describe
viajes de los llamados taxis verdes o SHL (Street-Hail Livery).

Variables destacadas:

VendorID: Empresa proveedora (ej. CMT, Curb, Myle).

lpep_pickup_datetime y lpep_dropoff_datetime: Inicio y fin del viaje.

RatecodeID: Tarifa aplicada (estándar, aeropuerto JFK/Newark, negociada, grupal, etc.).

passenger_count, trip_distance, fare_amount, tip_amount, tolls_amount.

payment_type: Tipo de pago (tarjeta, efectivo, etc.).

trip_type:Viaje tomado en la calle o despachado por llamada.

Incluye cargos adicionales como congestion_surcharge y desde 2025 el cbd_congestion_fee.

url_2 = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-06.parquet"
registros_viajes_eco = pd.read_parquet(url_2, engine="pyarrow")
