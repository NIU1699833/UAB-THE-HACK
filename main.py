from pandas import json_normalize
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
#guardar urls
url_ap = "https://raw.githubusercontent.com/albertgilopez/uabthehack-hackathon-2025/main/samples/anonymized/AP-info-v2-2025-06-13T14_45_01%2B02_00-ANON.json"
url_clients = "https://raw.githubusercontent.com/albertgilopez/uabthehack-hackathon-2025/main/samples/anonymized/client-info-2025-04-09T11_47_24%2B02_00-10487-ANON.json"
# Leer JSON
data_ap = requests.get(url_ap).json()
data_clients = requests.get(url_clients).json()
# Normalizar Access Points (con prefijo)
df_aps = json_normalize(
    data_ap,
    record_path=["radios"],
    meta=[
        "ap_deployment_mode",
        "client_count",
        "firmware_version",
        "group_name",
        "ip_address",
        "macaddr",
        "model",
        "name",
        "public_ip_address",
        "last_modified"
    ],
    meta_prefix="ap_",  # 👈 prefijo que evita conflictos
    errors='ignore'
)
# Clientes (estructura plana)
df_clients = pd.json_normalize(data_clients)

# Guardar resultados
df_aps.to_csv("access_points.csv", index=False)
df_clients.to_csv("clients.csv", index=False)

#guardar estadisticas
num_aps = df_aps['ap_macaddr'].nunique()
num_radios = len(df_aps)
num_devices = df_clients['macaddr'].nunique()
num_zonas = df_aps['ap_group_name'].nunique() if 'ap_group_name' in df_aps.columns else df_aps['ap_group_name'].count()

#buscar patrones hotspot
hotspots = (
    df_clients
    .groupby('group_name')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='num_dispositivos')
)


#picos temporales
df_clients['datetime'] = pd.to_datetime(df_clients['last_connection_time'], unit='ms', errors='coerce')
df_aps['datetime'] = pd.to_datetime(df_aps['ap_last_modified'], unit='s', errors='coerce')
df_clients['hora'] = df_clients['datetime'].dt.hour
df_clients['dia_semana'] = df_clients['datetime'].dt.day_name()
horas_pico = df_clients.groupby('hora').size().sort_values(ascending=False)
dias_pico = df_clients.groupby('dia_semana').size().sort_values(ascending=False)

#distribucion edificios
distribucion = (
    df_aps
    .groupby('ap_group_name')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='num_APs')
)

#mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Análisis WiFi - UAB Campus", fontsize=16)

#Hotspots
hotspots = df_clients.groupby('group_name').size().sort_values(ascending=False).head(10)
hotspots.plot(kind='bar', color='royalblue', ax=axes[0,0])
axes[0,0].set_title("Top 10 Zonas con Más Dispositivos")
axes[0,0].set_xlabel("Zona")
axes[0,0].set_ylabel("Nº de dispositivos")
axes[0,0].tick_params(axis='x', rotation=45)

#Conexiones por hora
df_clients['datetime'] = pd.to_datetime(df_clients['last_connection_time'], unit='ms', errors='coerce')
df_clients['hora'] = df_clients['datetime'].dt.hour
horas = df_clients.groupby('hora').size()
horas.plot(kind='bar', color='coral', ax=axes[0,1])
axes[0,1].set_title("Conexiones por Hora del Día")

#Días de la semana
df_clients['dia_semana'] = df_clients['datetime'].dt.day_name()
dias = df_clients.groupby('dia_semana').size().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
dias.plot(kind='bar', color='teal', ax=axes[1,0])
axes[1,0].set_title("Conexiones por Día de la Semana")
axes[1,0].tick_params(axis='x', rotation=45)

#Distribución de APs por grupo
dist = df_aps.groupby('ap_group_name').size().sort_values(ascending=False).head(10)
dist.plot(kind='barh', color='DarkOliveGreen', ax=axes[1,1])
axes[1,1].set_title("Distribución de APs por Grupo")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

df_aps = pd.read_csv("access_points.csv")
df_clients = pd.read_csv("clients.csv")

# Convertir timestamps
df_clients['datetime'] = pd.to_datetime(df_clients['last_connection_time'], unit='ms', errors='coerce')
df_clients['hora'] = df_clients['datetime'].dt.hour
df_clients['dia_semana'] = df_clients['datetime'].dt.day_name()

# A. Heatmap sobre el mapa del campus
if 'x' not in df_aps.columns or 'y' not in df_aps.columns:
    np.random.seed(42)
    df_aps['x'] = np.random.randint(50, 950, size=len(df_aps))
    df_aps['y'] = np.random.randint(50, 650, size=len(df_aps))

clientes_por_ap = df_clients.groupby('associated_device').size()
df_aps['num_clientes'] = df_aps['macaddr'].map(clientes_por_ap).fillna(0)

congestion = df_aps.groupby(['macaddr','band','channel'])['ap_client_count'].sum().reset_index()
congestion = congestion.sort_values('ap_client_count', ascending=False).head(15)

tipo_fabricante = df_clients.groupby(['client_type','manufacturer']).size().reset_index(name='count')
tipo_fabricante = tipo_fabricante.sort_values('count', ascending=False).head(15)

conexiones_hora = df_clients.groupby('hora').size()
prediccion = conexiones_hora.rolling(window=3, center=True).mean()

#Crear figura con subplots 2x2
fig, axes = plt.subplots(2, 2, figsize=(30, 24))
fig.suptitle("Dashboard Creativo: WiFi Campus UAB", fontsize=20)

# ---- Heatmap sobre el mapa del campus ----
img = plt.imread("mapa_uab.jpg")
axes[0,0].imshow(img, extent=[0,1000,0,700])

sc = axes[0,0].scatter(
    df_aps['x'], df_aps['y'],
    c=df_aps['num_clientes'],
    s=7,           # tamaño de los puntos
    cmap='hot',
    alpha=0.5,       # MÁS TRANSPARENTE
    edgecolors='black'
)
axes[0,0].set_title("Heatmap de Conexiones WiFi sobre el Campus", fontsize=16)
axes[0,0].set_xlabel("Coordenada X")
axes[0,0].set_ylabel("Coordenada Y")
plt.colorbar(sc, ax=axes[0,0], label='Número de clientes')

# ---- Congestión canales/banda ----
labels = congestion['macaddr'] + " " + congestion['band'].astype(str) + "GHz:" + congestion['channel'].astype(str)
axes[0,1].bar(labels, congestion['ap_client_count'], color='coral')
axes[0,1].set_title("Top APs por Clientes Conectados / Canal / Banda", fontsize=16)
axes[0,1].set_ylabel("Número de clientes")
axes[0,1].tick_params(axis='x', rotation=90)

# ---- Dispositivos por tipo/fabricante ----
labels_tf = tipo_fabricante['manufacturer'] + " (" + tipo_fabricante['client_type'] + ")"
axes[1,0].bar(labels_tf, tipo_fabricante['count'], color='mediumseagreen')
axes[1,0].set_title("Top Dispositivos por Tipo y Fabricante", fontsize=16)
axes[1,0].set_ylabel("Número de dispositivos")
axes[1,0].tick_params(axis='x', rotation=90)

# ---- Predicción picos por hora ----
axes[1,1].plot(conexiones_hora.index, conexiones_hora.values, marker='o', label='Conexiones reales')
axes[1,1].plot(prediccion.index, prediccion.values, marker='x', linestyle='--', color='red', label='Predicción simple')
axes[1,1].set_title("Predicción básica de picos por hora", fontsize=16)
axes[1,1].set_xlabel("Hora del Día")
axes[1,1].set_ylabel("Número de conexiones")
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_xticks(range(0,24))

#Ajustes finales
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()