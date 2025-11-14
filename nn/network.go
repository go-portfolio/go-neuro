package nn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)


// -----------------------------
// Сеть (Network)
// -----------------------------
type Network struct {
	Layers []Layer `json:"layers"`
}

// -----------------------------
// Конструктор сети
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
			In:      in,
			Out:     out,
			Weights: w,
			Biases:  b,
			Z:       make([]float64, out),
			A:       make([]float64, out),
		}
	}

	return &Network{Layers: layers}
}

// -----------------------------
// Функция активации
// -----------------------------
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// -----------------------------
// Forward pass
// -----------------------------
func (net *Network) ForwardFull(input []float64) ([]float64, [][]float64, [][]float64) {
	activations := [][]float64{input}
	zvals := [][]float64{}
	a := input

	for i := range net.Layers {
		layer := &net.Layers[i]
		z := make([]float64, layer.Out)
		a2 := make([]float64, layer.Out)

		for j := 0; j < layer.Out; j++ {
			sum := layer.Biases[j]
			for k := 0; k < layer.In; k++ {
				sum += layer.Weights[j][k] * a[k]
			}
			z[j] = sum
			a2[j] = sigmoid(sum)

			// сохраняем в слой
			layer.Z[j] = sum
			layer.A[j] = a2[j]
		}

		zvals = append(zvals, z)
		activations = append(activations, a2)
		a = a2
	}

	return a, activations, zvals
}

// Простое предсказание
func (net *Network) Predict(input []float64) []float64 {
	a := input
	for i := range net.Layers {
		layer := &net.Layers[i]
		next := make([]float64, layer.Out)
		for j := 0; j < layer.Out; j++ {
			sum := layer.Biases[j]
			for k := 0; k < layer.In; k++ {
				sum += layer.Weights[j][k] * a[k]
			}
			next[j] = sigmoid(sum)
			layer.Z[j] = sum
			layer.A[j] = next[j]
		}
		a = next
	}
	return a
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
		layer := &net.Layers[l]
		for i := 0; i < layer.Out; i++ {
			layer.Biases[i] -= lr * deltas[l][i]
			for j := 0; j < layer.In; j++ {
				layer.Weights[i][j] -= lr * deltas[l][i] * activations[l][j]
			}
		}
	}
}

// -----------------------------
// TrainBatch
// -----------------------------
func (net *Network) TrainBatch(samples [][]float64, targets [][]float64, epochs int, lr float64) {
	for e := 0; e < epochs; e++ {
		totalLoss := 0.0
		for idx, x := range samples {
			y := targets[idx]

			out, activations, zvals := net.ForwardFull(x)
			for i := range y {
				diff := out[i] - y[i]
				totalLoss += 0.5 * diff * diff
			}

			deltas := net.Backpropagate(y, activations, zvals)
			net.ApplyGradients(lr, deltas, activations)
		}

		if e%1000 == 0 || e == epochs-1 {
			fmt.Printf("Epoch %d, avg loss: %.6f\n", e, totalLoss/float64(len(samples)))
		}
	}
}

// -----------------------------
// Печать состояния слоя
// -----------------------------
func PrintLayerState(name string, L *Layer) {
	fmt.Printf("--- %s ---\n", name)
	fmt.Printf("Inputs: %d, Outputs: %d\n", L.In, L.Out)
	fmt.Println("Weights:")
	for i := range L.Weights {
		fmt.Println(L.Weights[i])
	}
	fmt.Println("Biases:")
	fmt.Println(L.Biases)
	fmt.Println("Z (w*x+b) and activations (A):")
	for i := range L.Z {
		fmt.Printf("Neuron %d: Z=%.4f, A=%.4f\n", i, L.Z[i], L.A[i])
	}
}
