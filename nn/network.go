package nn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Network struct {
	Layers []Layer `json:"layers"`
}

// -----------------------------
// Constructors
// -----------------------------

func NewNetwork(sizes []int) *Network {
	rand.Seed(time.Now().UnixNano())

	layers := make([]Layer, len(sizes)-1)

	for i := 0; i < len(sizes)-1; i++ {
		in := sizes[i]
		out := sizes[i+1]

		w := make([][]float64, out)
		b := make([]float64, out)

		for o := 0; o < out; o++ {
			b[o] = rand.NormFloat64() * 0.1
			w[o] = make([]float64, in)
			for j := 0; j < in; j++ {
				w[o][j] = rand.NormFloat64() * 0.1
			}
		}

		layers[i] = Layer{
			Weights: w,
			Biases:  b,
		}
	}

	return &Network{Layers: layers}
}

// -----------------------------
// Activation
// -----------------------------

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// -----------------------------
// Forward
// -----------------------------

func (net *Network) Predict(input []float64) []float64 {
	a := input

	for _, L := range net.Layers {
		next := make([]float64, len(L.Biases))
		for i := range next {
			sum := L.Biases[i]
			for j := range a {
				sum += L.Weights[i][j] * a[j]
			}
			next[i] = sigmoid(sum)
		}
		a = next
	}

	return a
}

// Возвращает output + все активации + z-values
func (net *Network) ForwardFull(input []float64) ([]float64, [][]float64, [][]float64) {
	activations := [][]float64{input}
	zvals := [][]float64{}
	a := input

	for _, L := range net.Layers {
		z := make([]float64, len(L.Biases))
		a2 := make([]float64, len(L.Biases))

		for i := range z {
			sum := L.Biases[i]
			for j := range a {
				sum += L.Weights[i][j] * a[j]
			}
			z[i] = sum
			a2[i] = sigmoid(sum)
		}

		zvals = append(zvals, z)
		activations = append(activations, a2)
		a = a2
	}

	return a, activations, zvals
}

// -----------------------------
// Backpropagation
// -----------------------------

func (net *Network) Backpropagate(target []float64, activations [][]float64, zvals [][]float64) [][]float64 {
	L := len(net.Layers)
	deltas := make([][]float64, L)

	// Выходной слой
	last := L - 1
	outAct := activations[last+1]
	outZ := zvals[last]

	deltas[last] = make([]float64, len(outAct))
	for i := range deltas[last] {
		deltas[last][i] = (outAct[i] - target[i]) * sigmoidPrime(outZ[i])
	}

	// Скрытые слои
	for l := L - 2; l >= 0; l-- {
		nextLayer := net.Layers[l+1]

		z := zvals[l]
		deltas[l] = make([]float64, len(z))

		for i := range z {
			sum := 0.0
			for k := range nextLayer.Biases {
				sum += nextLayer.Weights[k][i] * deltas[l+1][k]
			}
			deltas[l][i] = sum * sigmoidPrime(z[i])
		}
	}

	return deltas
}

// -----------------------------
// Apply gradients
// -----------------------------

func (net *Network) ApplyGradients(lr float64, deltas [][]float64, activations [][]float64) {
	for l := range net.Layers {
		L := &net.Layers[l]

		for i := range L.Biases {
			L.Biases[i] -= lr * deltas[l][i]
		}

		for i := range L.Weights {
			for j := range L.Weights[i] {
				L.Weights[i][j] -= lr * deltas[l][i] * activations[l][j]
			}
		}
	}
}

// -----------------------------
// Train Batch
// -----------------------------

func (net *Network) TrainBatch(samples [][]float64, targets [][]float64, epochs int, lr float64) {
	for e := 0; e < epochs; e++ {
		totalLoss := 0.0
		for idx, x := range samples {
			y := targets[idx]
			// прямой проход
			out, activations, zvals := net.ForwardFull(x)

			// подсчет ошибки
			for i := range y {
				diff := out[i] - y[i]
				totalLoss += 0.5 * diff * diff
			}

			// обратное распространение
			deltas := net.Backpropagate(y, activations, zvals)

			// применение градиентов
			net.ApplyGradients(lr, deltas, activations)
		}

		if e%1000 == 0 || e == epochs-1 {
			avgLoss := totalLoss / float64(len(samples))
			fmt.Printf("Epoch %d, avg loss: %.6f\n", e, avgLoss)
		}
	}
}
