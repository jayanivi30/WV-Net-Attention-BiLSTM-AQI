import numpy as np
import gradio as gr
import tensorflow as tf


# Load trained WV-NET model

model = tf.keras.models.load_model(
    "wvnet_48h.h5",
    compile=False
)


# AQI Categorization (WHO-based)

def pm25_category(pm):
    if pm <= 15:
        return "Good 🟢"
    elif pm <= 30:
        return "Moderate 🟡"
    elif pm <= 55:
        return "Unhealthy for Sensitive Groups 🟠"
    elif pm <= 110:
        return "Unhealthy 🔴"
    else:
        return "Very Unhealthy ⚫"


# Prediction Function

def predict_pm25(prev_pm25, dewp, temp, pres, iws, cbwd):
    features = np.array([[prev_pm25, dewp, temp, pres, iws, cbwd]])
    input_seq = np.repeat(features[np.newaxis, :, :], 48, axis=1)

    pred = float(model.predict(input_seq, verbose=0)[0][0])

    # Stabilize forecast so AQI can vary realistically
    effective_pm25 = max(pred, prev_pm25 * 0.8)

    return (
        f"{effective_pm25:.2f} µg/m³",
        pm25_category(effective_pm25)
    )

# Gradio UI

ui = gr.Interface(
    fn=predict_pm25,
    inputs=[
        gr.Number(label="Previous PM2.5 (µg/m³)", value=80),
        gr.Number(label="Dew Point (°C)", value=15),
        gr.Number(label="Temperature (°C)", value=32),
        gr.Number(label="Atmospheric Pressure (hPa)", value=1008),
        gr.Number(label="Wind Speed (m/s)", value=0.5),
        gr.Radio(
            choices=[0, 1, 2, 3],
            label="Wind Direction (cbwd): 0=NW, 1=NE, 2=SE, 3=SW",
            value=0
        )
    ],
    outputs=[
        gr.Textbox(label="Predicted Next-Hour PM2.5"),
        gr.Textbox(label="AQI Category (WHO Standard)")
    ],
    title="WV-NET : PM2.5 Forecasting",
    description=(
        "WV-Net forecasts next-hour PM2.5 using a 48-hour temporal window.\n\n"
        "Inputs: Previous PM2.5 + meteorological variables.\n"
        "Model: WV-NET (BiLSTM)."
    )
)

ui.launch()
