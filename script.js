const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compilar el modelo
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// Función para entrenar el modelo
async function entrenar() {
    const x = tf.tensor2d([[-6], [-5], [-4], [-3], [-2], [-1], [0], [1], [2]]);
    const y = tf.tensor2d([[-6], [-4], [-2], [0], [2], [4], [6], [8], [10]]);

    const surface = { name: 'Loss', tab: 'Training' };
    const history = [];

    await model.fit(x, y, {
        epochs: 250,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                history.push(log);
                tfvis.show.history(surface, history, ['loss']);
            }
        }
    });
}

// Función para predecir
function predecir() {
    const x_value = parseFloat(document.getElementById('x_value').value);
    const prediccion = model.predict(tf.tensor2d([[x_value]]));
    prediccion.data().then((data) => {
        document.getElementById('resultado').innerText = `El resultado de predecir para ${x_value} es: ${data[0]}`;
    });
}