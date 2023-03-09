import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sklearn


# carga dataFrame
df_modelo = pd.read_csv('model/df_modelo.csv')
#borro variable a predecir
df_modelo = df_modelo.drop("UNITPRICE_DEF", axis=1)
# Carga del modelo entrenado
best_xgb_model = joblib.load('model/best_xgb_model.pkl')

# Lista de columnas a mostrar en la interfaz
columnas_interfaz = ['ROOMNUMBER', 'CONSTRUCTEDAREA', 'LEADS_ASKING', 'NET_HOUSEHOLD_INCOME', 'EFFORTRATE_SALE', 'UNITPRICE_ASKING', 'CADASTRALQUALITYID']

# Filtrar las columnas del dataframe para mostrar solo las columnas seleccionadas
df_interfaz = df_modelo[columnas_interfaz]


# Definición de la función para hacer la predicción
def predict_price(df_modelo, best_xgb_model):
    nueva_casa = pd.DataFrame({'amenity_1': [df_modelo['amenity_1']],
                               'amenity_2': [df_modelo['amenity_2']],
                               'amenity_3': [df_modelo['amenity_3']],
                               '201803': [df_modelo['201803']],
                               '201806': [df_modelo['201806']],
                               '201809': [df_modelo['201809']],
                               '201812': [df_modelo['201812']],
                               'CONSTRUCTEDAREA': [df_modelo['CONSTRUCTEDAREA']],
                               'ROOMNUMBER': [df_modelo['ROOMNUMBER']],
                               'BATHNUMBER': [df_modelo['BATHNUMBER']],
                               'HASTERRACE': [df_modelo['HASTERRACE']],
                               'HASLIFT': [df_modelo['HASLIFT']],
                               'HASAIRCONDITIONING': [df_modelo['HASAIRCONDITIONING']],
                               'HASPARKINGSPACE': [df_modelo['HASPARKINGSPACE']],
                               'HASNORTHORIENTATION': [df_modelo['HASNORTHORIENTATION']],
                               'HASSOUTHORIENTATION': [df_modelo['HASSOUTHORIENTATION']],
                               'HASEASTORIENTATION': [df_modelo['HASEASTORIENTATION']],
                               'HASWESTTORIENTATION': [df_modelo['HASWESTORIENTATION']],
                               'HASBOXROOM': [df_modelo['HASBOXROOM']],
                               'HASWARDROBE': [df_modelo['HASWARDROBE']],
                               'HASSWIMMINGPOOL': [df_modelo['HASSWIMMINGPOOL']],
                               'HASDOORMAN': [df_modelo['HASDOORMAN']],
                               'HASGARDEN': [df_modelo['HASGARDEN']],
                               'ISDUPLEX': [df_modelo['ISDUPLEX']],
                               'ISSTUDIO': [df_modelo['ISSTUDIO']],
                               'ISINTOPFLOOR': [df_modelo['ISINTOPFLOOR']],
                               'FLOORCLEAN_x': [df_modelo['FLOORCLEAN_x']],
                               'FLATLOCATIONID_x': [df_modelo['FLATLOCATIONID_x']],
                               'CADCONSTRUCTIONYEAR': [df_modelo['CADCONSTRUCTIONYEAR']],
                               'CADASTRALQUALITYID': [df_modelo['CADASTRALQUALITYID']],
                               'BUILTTYPEID_1': [df_modelo['BUILTTYPEID_1']],
                               'BUILTTYPEID_2': [df_modelo['BUILTTYPEID_2']],
                               'BUILTTYPEID_3': [df_modelo['BUILTTYPEID_3']],
                               'DISTANCE_TO_CITY_CENTER': [df_modelo['DISTANCE_TO_CITY_CENTER']],
                               'DISTANCE_TO_METRO': [df_modelo['DISTANCE_TO_METRO']],
                               'DISTANCE_TO_CASTELLANA': [df_modelo['DISTANCE_TO_CASTELLANA']],
                               'FLOORCLEAN_FILLED': [df_modelo['FLOORCLEAN_FILLED']],
                               'FLATLOCATIONID_FILLED': [df_modelo['FLATLOCATIONID_FILLED']],
                               'UNITPRICE_ASKING': [df_modelo['UNITPRICE_ASKING']],
                               'LEADS_ASKING': [df_modelo['LEADS_ASKING']],
                               'NET_HOUSEHOLD_INCOME': [df_modelo['NET_HOUSEHOLD_INCOME']],
                               'EFFORTRATE_SALE': [df_modelo['EFFORTRATE_SALE']]})

    precio = best_xgb_model.predict(nueva_casa)
    precio_vivienda = precio[0] * nueva_casa['CONSTRUCTEDAREA'][0]
    precio_barrio = nueva_casa['UNITPRICE_ASKING'][0]

    return precio[0][0]

# Configuración de la interfaz de Streamlit
st.title('Predicción del precio de una casa')
st.write('Ingresa los detalles de la casa para obtener una predicción de su precio.')

col1, col2 = st.columns(2)

idx = col1.number_input('Índice', min_value=0, max_value=10, step=1, value=0)
room_number = col1.number_input('Número de habitaciones', min_value=1, max_value=5, step=1, value=3)
constructed_area = col1.number_input('Área construida (m²)', min_value=30, max_value=400, step=1, value=150)
leads_asking = col1.number_input('Leads preguntando', min_value=0.0, max_value=50.0, step=0.1, value=2.1)
net_household_income = col1.number_input('Ingreso neto del hogar', min_value=20000, max_value=1000000, step=100, value=35000)
effort_rate_sale = col1.slider('Tasa de esfuerzo de venta', min_value=0.10, max_value=1.0, step=0.01, value=0.28)
unitprice_asking = col1.number_input('Precio unitario de oferta (USD/m²)', min_value=500, max_value=10000, step=10, value=2500)
cadastral_quality_id = col1.number_input('ID de calidad catastral', min_value=1, max_value=9, step=1, value=3)

# Creación del DataFrame con los valores ingresados
input_data = {'ROOMNUMBER': room_number,
              'CONSTRUCTEDAREA': constructed_area,
              'LEADS_ASKING': leads_asking,
              'NET_HOUSEHOLD_INCOME': net_household_income,
              'EFFORTRATE_SALE': effort_rate_sale,
              'UNITPRICE_ASKING': unitprice_asking,
              'CADASTRALQUALITYID': cadastral_quality_id}
input_df_modelo = pd.DataFrame(input_data, index=[0])

def predict_price(df_modelo, idx, input_data, best_xgb_model):
    selected_cols = ['amenity_1','amenity_2','amenity_3','201803','201806','201809','201812','CONSTRUCTEDAREA',
                     'ROOMNUMBER','BATHNUMBER','HASTERRACE','HASLIFT','HASAIRCONDITIONING','HASPARKINGSPACE',
                     'HASNORTHORIENTATION','HASSOUTHORIENTATION','HASEASTORIENTATION','HASWESTORIENTATION','HASBOXROOM',
                     'HASWARDROBE','HASSWIMMINGPOOL','HASDOORMAN','HASGARDEN','ISDUPLEX','ISSTUDIO','ISINTOPFLOOR','FLOORCLEAN_x',
                     'FLATLOCATIONID_x','CADCONSTRUCTIONYEAR','CADASTRALQUALITYID','BUILTTYPEID_1','BUILTTYPEID_2','BUILTTYPEID_3',
                     'DISTANCE_TO_CITY_CENTER','DISTANCE_TO_METRO','DISTANCE_TO_CASTELLANA','FLOORCLEAN_FILLED','FLATLOCATIONID_FILLED',
                     'UNITPRICE_ASKING','LEADS_ASKING','NET_HOUSEHOLD_INCOME','EFFORTRATE_SALE']
    #
    # Select a single instance
    #
    input_vars = df_modelo.iloc[[idx]][selected_cols]
  
    #
    # Overwrite parameters
    # 
    
    for feature_name in input_data:
      print('Overwriting: ' +  feature_name + ' -> ' + str(input_data[feature_name]))
      input_vars.loc[idx, feature_name] = input_data[feature_name]

    prediction = best_xgb_model.predict(input_vars)[0]
    return prediction


# Ejecución de la función para obtener la predicción
if col1.button('Obtener predicción'):
    #idx = 0 # Record index
    df = pd.DataFrame([[40.416729, -3.703339]], columns=['lat', 'lon'])
    col2.write('Mapa de situación')
    col2.map(df)
    predicted_price = predict_price(df_modelo, idx, input_data, best_xgb_model)
    min_price = predicted_price * 0.8
    max_price = predicted_price * 1.1
    st.write('El precio de la casa es:', str(round(constructed_area * predicted_price)) + ' € -> ' + str(predicted_price) + ' €/m²')
    st.write('El intervalo de confianza es: [',str(min_price),' - ', str(max_price) + ']')