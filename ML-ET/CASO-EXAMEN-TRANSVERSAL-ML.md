Intrucciones:  
Eres un experto en Data Science and Data Engieniere con conocimientos extensos en machine learning y algoritmos de aprendizaje automatico, te has desarrollado con larga trayectoria en el rubro de las app de Drivers como uber, Didi , Etc. necesito que tu amplio conocimiento lo apliques en este proyecto. Utilizaremos metodologia CRISP-DM, con herramientas como Google collab y lenguajes Python para la estadistica y codigos de Ml.

Caso de estudio: Optimización del transporte en Nueva York:

La Comisión de Taxis y Limusinas de Nueva York (TLC) dispone de registros masivos de viajes de taxis amarillos, taxis verdes, vehículos de alquiler (FHV) y vehículos de alto volumen (HVFHV). Se busca analizar estos datos para mejorar la eficiencia del transporte, la experiencia del pasajero y la planificación urbana.

El análisis de estos datasets permite estudiar patrones de movilidad, optimizar recursos, anticipar la demanda y detectar comportamientos anómalos, ofreciendo un enfoque integral de gestión de transporte urbano.

Problemáticas que se pueden abordar con los datasets

### EXAMEN TRANSVERSAL - ET

**2.**    **Segmentación de zonas de la ciudad (No
supervisado – Clustering)**

Problema:

Algunas zonas tienen alta concentración de viajes en ciertas horas, mientras que otras permanecen con poca actividad. Esto dificulta la distribución eficiente de taxis y vehículos de alquiler.

Objetivo:

Agrupar zonas de la ciudad según volumen de viajes, hora pico y tipo de servicio.

Identificar áreas con alta demanda o saturadas.

Optimizar la asignación de flotas y planificación de estaciones de vehículos.

### Dataset

aplicable: Yellow Taxi, Green Taxi, FHV / HVFHV

**2**.    High Volume FHV Trip Records (HVFHS) 
Viajes de empresas de alto volumen (más de 10.000 viajes diarios): Uber, Lyft, Via, Juno.

Variables clave:
hvfhs_license_num: Licencia de la empresa (Uber, Lyft, etc.).
request_datetime, pickup_datetime, dropoff_datetime.

trip_miles, trip_time.
Costos: base_passenger_fare, tolls, sales_tax, congestion_surcharge, airport_fee.
tips, driver_pay.
Flags: shared_request_flag / shared_match_flag → si se solicitó o efectivamente se compartió el viaje.
access_a_ride_flag → viajes gestionados por la MTA.
wav_request_flag y wav_match_flag → accesibilidad para silla de ruedas.
Incluye también el cbd_congestion_fee desde 2025.

url_4 = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-06.parquet"
Registros_vehiculos_gran_volumen = pd.read_parquet(url_4, engine="pyarrow")
