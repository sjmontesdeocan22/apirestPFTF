import torch
import pickle

MODEL_PATH='app/autoencoder_model_scripted_v1.pt'
model = torch.jit.load(MODEL_PATH)
model.eval()
umbral=0.40

def leer_dato(uploaded_file):
    dato = pickle.loads(uploaded_file.getvalue())

    return dato

def predecir(dato):
    fn_perdida = torch.nn.L1Loss(reduction='none')
    reconstrucciones = model(torch.from_numpy(dato).float())
    perdida = fn_perdida(reconstrucciones, torch.from_numpy(dato).float()).mean()

    return torch.lt(perdida, umbral)

def obtener_categoria(comparaciones):
    if comparaciones.item():
        categoria = 'Normal'
    else:
        categoria = 'Anormal'

    return categoria




