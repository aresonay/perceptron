import streamlit as st
from perceptron import Neuron
import numpy as np





input_values = []
weight_values = []


def inputs_cols_generator(number_inputs, name):
    values = st.columns(number_inputs)
    w_x = []
    for i in range(number_inputs):
        w_x.append(values[i].number_input(f'${name}_{i}$', key=f'{name}_{i}'))
        if name == 'w':
            weight_values.append(w_x[i])
        else:
            input_values.append(w_x[i])
        
    st.code(w_x)    
        

        

style = f'''
    <style>
        .appview-container .main .block-container {{
            max-width: 90%;
        }}
    </style>
  
  
  '''

st.markdown(style, unsafe_allow_html=True)

       

st.title("Simulador de neurona")

image = Image.open('neurona_fp_1000.jpg')

st.image(image, caption='Ilustración de una Neurona')


st.write("Elige el número de entradas/pesos que tendrá la neurona")
number_inputs_weights = st.slider(
    'number_inputs',
    1,10, 
    label_visibility="collapsed",
    key='number_inputs_weights'
)

st.write(number_inputs_weights)
st.subheader("Pesos")
w_values = inputs_cols_generator(number_inputs_weights, 'w')
st.subheader("Entradas")
x_values = inputs_cols_generator(number_inputs_weights, 'x')
st.write(x_values)






col_st_1, col_st_2 = st.columns(2)

with col_st_1:
    st.subheader("Sesgo")
    bias = st.number_input("Introduce el valor del sesgo" ,key="bias")
    st.write(bias)

with col_st_2:
    st.subheader("Función de activación")
    activation_function = st.selectbox(
        "Función de activación", 
        ("relu", "tanh", "sigmoid", "binarystep", "linear")
    )
    st.write(activation_function)


button_output = st.button(label="Calcular la salida", key="output_button")


if button_output:
    perceptron = Neuron(weights=weight_values, bias=bias, func=activation_function)
    st.write(f'La salida es: {perceptron.run(input_data=input_values)}')
    
 

  



