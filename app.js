'use strict';

/*
	Linear Regression with TensorFlowJS
*/

// create the Arays that will hold our values
let x_vals = [];
let y_vals = [];

let m, b;

// define the learning rate and the optimizer
const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

// ceate the canvas
function setup() {
	createCanvas(640, 480);
	// y = mx + b
	m = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));

}

// The Loss function
function loss(pred, labels) {
	return pred.sub(labels).square().mean();
}

// predictor function
function predict(x) {
	// if array make into a tensor
	const xs = tf.tensor1d(x);
	// y = mx + b
	const ys = xs.mul(m).add(b);
	return ys;
}

// event handler
function mousePressed() {
	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, 0, height, 1, 0);

	x_vals.push(x);
	y_vals.push(y);
}

// add a draw loop
function draw() {

	// lets put all these inside a tidy function
	tf.tidy(() => {
		if (x_vals.length > 0) {
			// make a tensor ut of the y_vals
			const ys = tf.tensor1d(y_vals);
			// Train the model.
		  optimizer.minimize(() => loss(predict(x_vals), ys));
		}
	})


	// create and render the canvas on the page
	background(0);

	stroke(255);
	strokeWeight(4);

	for (let i = 0; i < x_vals.length; i++){
		let px = map(x_vals[i], 0, 1, 0, width);
		let py = map(y_vals[i], 0, 1, height, 0);
		point(px, py);
	}

	tf.tidy(() => {
		const xs = [0, 1];
		const ys = predict(xs);
		let lineY = ys.dataSync();

		// draw a line
		let x1 = map(xs[0], 0, 1, 0, width);
		let x2 = map(xs[1], 0, 1, 0, width);

		let y1 = map(lineY[0], 0, 1, height, 0);
		let y2 = map(lineY[1], 0, 1, height, 0);

		line(x1, y1, x2, y2);
	})
}


/*

Some ideas for extending this example:

make it moreinteractive and have some Ui for the main parameters, like the learning rate, stroke color, strokeWeight, etc

and print to the page things like the loss data as a graph, etc

then after adding more things add to my portfolio site

*/