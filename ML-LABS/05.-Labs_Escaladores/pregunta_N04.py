# # PREGUNTA 4: ¿Cuál escalador recomendarías si tu dataset tiene muchos outliers? ¿Y si necesitas que tus datos estén entre [0,1]?

# In[28]:


# ==================================================
# EVIDENCIA PREGUNTA 4: Comparacion de escaladores con outliers y necesidad de rango [0,1]
# ==================================================

print("=" * 80)
print("ELECCION DE ESCALADORES: OUTLIERS vs RANGO [0,1]")
print("=" * 80)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Crear datos sinteticos con outliers extremos
np.random.seed(42)
n_samples = 1000

print("\nEXPERIMENTO: Datos con outliers extremos")
print("=" * 60)

# Crear dataset con outliers
normal_data = np.random.normal(50, 10, int(n_samples * 0.9))
outliers = np.random.uniform(200, 300, int(n_samples * 0.1))
data_with_outliers = np.concatenate([normal_data, outliers])

# Crear DataFrame para analisis
df_outliers = pd.DataFrame({
    'variable_normal': np.random.normal(100, 15, n_samples),
    'variable_con_outliers': data_with_outliers,
    'variable_pequena': np.random.uniform(1, 5, n_samples)
})

print("ANALISIS DE OUTLIERS:")
for col in df_outliers.columns:
    Q1 = df_outliers[col].quantile(0.25)
    Q3 = df_outliers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = ((df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)).sum()
    outliers_pct = outliers_count / len(df_outliers) * 100

    print(f"   {col:20s}: {outliers_count:3d} outliers ({outliers_pct:5.1f}%)")
    print(f"     Rango: [{df_outliers[col].min():6.1f} - {df_outliers[col].max():6.1f}]")

# Preparar escaladores
escaladores = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(n_quantiles=min(1000, n_samples), random_state=42)
}

# Aplicar escaladores
datos_escalados = {}
for nombre, escalador in escaladores.items():
    datos_escalados[nombre] = pd.DataFrame(
        escalador.fit_transform(df_outliers),
        columns=df_outliers.columns
    )

print(f"\nCOMPARACION DE ESCALADORES:")
print("=" * 60)

# Analizar impacto en variable con outliers
var_outliers = 'variable_con_outliers'

print(f"\nANALISIS DE '{var_outliers}':")
print(f"   Original: min={df_outliers[var_outliers].min():.1f}, max={df_outliers[var_outliers].max():.1f}")

for nombre, datos in datos_escalados.items():
    min_val = datos[var_outliers].min()
    max_val = datos[var_outliers].max()
    std_val = datos[var_outliers].std()

    # Verificar si esta en rango [0,1]
    en_rango_01 = (min_val >= -0.001) and (max_val <= 1.001)
    marca_rango = "SI" if en_rango_01 else "NO"

    print(f"   {nombre:18s}: [{min_val:6.3f} - {max_val:6.3f}] std={std_val:.3f} [0,1]={marca_rango}")

# Visualizar el efecto en histogramas
print(f"\nVISUALIZACION DEL IMPACTO DE OUTLIERS:")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Original
axes[0].hist(df_outliers[var_outliers], bins=50, alpha=0.7, color='red', edgecolor='black')
axes[0].set_title('Original (con outliers)', fontweight='bold')
axes[0].set_xlabel('Valor')
axes[0].set_ylabel('Frecuencia')

# Escaladores
for i, (nombre, datos) in enumerate(datos_escalados.items(), 1):
    axes[i].hist(datos[var_outliers], bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{nombre}', fontweight='bold')
    axes[i].set_xlabel('Valor escalado')
    axes[i].set_ylabel('Frecuencia')

    # Agregar lineas en 0 y 1 si esta en ese rango
    if datos[var_outliers].min() >= -0.1 and datos[var_outliers].max() <= 1.1:
        axes[i].axvline(x=0, color='green', linestyle='--', alpha=0.7, label='0')
        axes[i].axvline(x=1, color='green', linestyle='--', alpha=0.7, label='1')
        axes[i].legend()

# Ocultar el ultimo subplot si no se usa
if len(axes) > len(datos_escalados) + 1:
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nRECOMENDACIONES ESPECIFICAS:")
print("=" * 60)

print(f"\nPARA DATASETS CON MUCHOS OUTLIERS:")
print(f"   RECOMENDADO: RobustScaler")
print(f"      Usa mediana e IQR (menos sensible a outliers)")
print(f"      Mantiene la forma de la distribucion central")
print(f"      No garantiza rango especifico")
print(f"   ")
print(f"   ALTERNATIVA: QuantileTransformer")
print(f"      Transforma a distribucion uniforme")
print(f"      Muy robusto a outliers extremos")
print(f"      Puede alterar la forma de la distribucion")
print(f"   ")
print(f"   EVITAR: MinMaxScaler")
print(f"      Muy sensible a outliers")
print(f"      Un solo outlier puede comprimir todos los demas valores")

print(f"\nPARA NECESIDAD DE RANGO [0,1]:")
print(f"   RECOMENDADO: MinMaxScaler")
print(f"      Garantiza exactamente el rango [0,1]")
print(f"      Mantiene proporciones relativas")
print(f"      Cuidado con outliers")
print(f"   ")
print(f"   ALTERNATIVA: QuantileTransformer(output_distribution='uniform')")
print(f"      Garantiza rango [0,1]")
print(f"      Robusto a outliers")
print(f"      Cambia la distribucion a uniforme")

print(f"\nCASO ESPECIFICO: OUTLIERS + NECESIDAD DE [0,1]:")
print(f"   ESTRATEGIA HIBRIDA:")

print(f"\n   OPCION A: Tratamiento de outliers + MinMaxScaler")
print(f"   1. Identificar y tratar outliers (clip, remove, o winsorize)")
print(f"   2. Aplicar MinMaxScaler al dataset limpio")
print(f"   3. Garantiza rango [0,1] sin distorsion por outliers")

print(f"\n   OPCION B: QuantileTransformer uniforme")
print(f"   1. Aplicar QuantileTransformer(output_distribution='uniform')")
print(f"   2. Automaticamente robusto y en rango [0,1]")
print(f"   3. Cambia la distribucion original")

print(f"\n   OPCION C: RobustScaler + normalizacion manual")
print(f"   1. Aplicar RobustScaler (robusto a outliers)")
print(f"   2. Normalizar manualmente: (x - min) / (max - min)")
print(f"   3. Mas pasos, pero control total")

# Demostracion practica con Boston Housing
print(f"\nAPLICACION A BOSTON HOUSING:")
print("=" * 60)

# Verificar outliers en Boston Housing
print(f"OUTLIERS EN BOSTON HOUSING:")
for col in X.columns[:5]:  # Solo primeras 5 columnas para brevedad
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    outliers_pct = outliers_count / len(X) * 100

    if outliers_count > 0:
        print(f"   {col:8s}: {outliers_count:2d} outliers ({outliers_pct:4.1f}%)")

# Recomendacion especifica para Boston Housing
columnas_con_outliers = []
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    if outliers_count > 0:
        columnas_con_outliers.append(col)

print(f"\nRECOMENDACION PARA BOSTON HOUSING:")
if len(columnas_con_outliers) > 0:
    print(f"   {len(columnas_con_outliers)} variables tienen outliers: {columnas_con_outliers[:3]}...")
    print(f"   Para KNN/SVM: usar RobustScaler")
    print(f"   Para redes neuronales: QuantileTransformer + output='uniform'")
    print(f"   Evitar MinMaxScaler sin tratamiento de outliers")
else:
    print(f"   No se detectaron outliers significativos")
    print(f"   MinMaxScaler es seguro para rango [0,1]")
    print(f"   StandardScaler es buena opcion general")

print("\n" + "=" * 80)


# # RESUMEN EJECUTIVO DEL TALLER
# 
# ## Respuestas Completas con Evidencia de Ejecución
# 
# ### PREGUNTA 1: KNN vs. Regresión Lineal
# **Respuesta:** KNN mejora significativamente con escalado porque usa distancia euclidiana, donde variables con rangos grandes dominan el cálculo. La regresión lineal es menos sensible porque ajusta automáticamente los coeficientes β a la escala de cada variable.
# 
# **Evidencia:** En nuestro experimento, KNN mejoró ~30-40% con escalado, mientras que regresión lineal mostró cambios mínimos (<5%).
# 
# ### PREGUNTA 2: Situaciones Prácticas Críticas
# **Respuesta:** La escala es crítica en:
# - **Salud:** Diagnósticos automáticos, dosificación
# - **Finanzas:** Detección de fraude, evaluación crediticia
# - **Seguridad:** Sistemas biométricos, detección de intrusiones
# - **Transporte:** Vehículos autónomos, optimización
# 
# **Evidencia:** Simulamos datos de e-commerce, detección de fraude y diagnóstico médico, mostrando cómo diferentes rangos (ingresos ~$180k vs compras ~15) distorsionan algoritmos basados en distancia.
# 
# ### PREGUNTA 3: Riesgos del Escalado Incorrecto
# **Respuesta:** El principal riesgo es **data leakage**: usar estadísticas del test set para escalar, causando estimaciones optimistas del rendimiento.
# 
# **Evidencia:** Demostramos que escalar antes de dividir puede mostrar mejoras artificiales. Los parámetros del escalador (media, std) "filtran" información del test set al modelo.
# 
# ### PREGUNTA 4: Escaladores según Contexto
# **Respuesta:** 
# - **Con outliers:** RobustScaler (usa mediana e IQR)
# - **Necesidad de [0,1]:** MinMaxScaler o QuantileTransformer
# - **Ambos casos:** QuantileTransformer con output uniforme
# 
# **Evidencia:** Creamos datos sintéticos con 10% de outliers extremos y comparamos 4 escaladores, mostrando cómo MinMaxScaler se distorsiona mientras RobustScaler mantiene estabilidad.
# 
# ## Conclusiones Principales
# 
# 1. **Escalado es crítico** para algoritmos basados en distancia (KNN, SVM, clustering)
# 2. **Orden importa:** Siempre dividir datos ANTES de escalar para evitar data leakage
# 3. **Contexto define la elección:** Outliers → RobustScaler, Rango [0,1] → MinMaxScaler
# 4. **Validación es esencial:** Usar Pipeline para automatizar y evitar errores
# 
# ## Metodología Aplicada
# 
# - **Análisis exploratorio** completo del dataset Boston Housing  
# - **Experimentos controlados** comparando escaladores  
# - **Visualizaciones** del impacto en distribuciones  
# - **Métricas cuantitativas** (RMSE) para validar hipótesis  
# - **Simulaciones** de casos reales con diferentes características  
# - **Buenas prácticas** implementadas y validadas

# EJEMPLOS DE USO DE ESCALADORES EN PYTHON (BostonHousing)
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Cargar datos (asegúrate de tener BostonHousing.csv en el mismo directorio)
df = pd.read_csv('BostonHousing.csv')
X = df.drop(columns=['medv'])  # Usamos solo las variables predictoras

print('--- Ejemplo: StandardScaler ---')
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print('Media después de escalar (debería ser ~0):', X_std.mean(axis=0)[:3])
print('Desviación estándar (debería ser ~1):', X_std.std(axis=0)[:3])

print('\n--- Ejemplo: MinMaxScaler ---')
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
print('Mínimo después de escalar:', X_mm.min(axis=0)[:3])
print('Máximo después de escalar:', X_mm.max(axis=0)[:3])

print('\n--- Ejemplo: RobustScaler ---')
scaler_rob = RobustScaler()
X_rob = scaler_rob.fit_transform(X)
print('Mediana después de escalar (debería ser ~0):', pd.DataFrame(X_rob).median().values[:3])
print('IQR después de escalar (debería ser ~1):', (pd.DataFrame(X_rob).quantile(0.75) - pd.DataFrame(X_rob).quantile(0.25)).values[:3])

print('\n--- Ejemplo: Normalizer (L2 por fila) ---')
scaler_norm = Normalizer(norm='l2')
X_norm = scaler_norm.fit_transform(X)
# La suma de los cuadrados de cada fila debe ser 1
print('Norma L2 de la primera fila:', (X_norm[0]**2).sum())

# Nota: Para la mayoría de los modelos de ML, StandardScaler o RobustScaler son los más usados.
# MinMaxScaler es útil si necesitas rango [0,1]. Normalizer se usa en casos especiales (texto, distancias angulares).
