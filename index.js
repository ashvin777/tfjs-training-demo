/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataReq = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  );
  const carsData = await carsDataReq.json();
  const cleaned = carsData
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower
    }))
    .filter(car => car.mpg != null && car.horsepower != null);

  return cleaned;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));
  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
  // More code will be added below

  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);

function createModel() {
  // Create a sequential model - create multiple layers executed sequentially
  const model = tf.sequential();

  // tranformations layers

  // Add a single hidden layer - inputShape - shape of tensor ( 1-1 mapping, ) inputShape 1 [x y] 2d is tensor
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

//Is all model are trained in same fashion?? clustering models regresssion etc ?? Clasification
/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_ (to get different results each time of trainignin)
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower); //inputs
    const labels = data.map(d => d.mpg); //trained data

    console.log(data, labels);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]); //output expected range 0-1 (score?)
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling for probability
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin)); // 1/3 or 1/2 0-1 range

    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    console.log(normalizedInputs, normalizedLabels);

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later., these are passed to denormalized, basically the demulitply
      inputMax,
      inputMin,
      labelMax,
      labelMin
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.

  //to tell how to traine the model
  model.compile({
    optimizer: tf.train.adam(), //predefine algo, to decide compile the models
    loss: tf.losses.meanSquaredError, //calculating loss predefined method
    metrics: ['mse'] //
  });

  const batchSize = 32; //no of inputs passed at once, while training model (shuffled data from above will be passed)
  const epochs = 50; //no of times model will be trained using same data but shuffled

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      //results of model training output
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    //denormalized with max and min values
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted']
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}
