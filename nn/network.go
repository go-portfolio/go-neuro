package nn

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

// ===============================
// Структура сети
// ===============================

type Network struct {
	Layers []Layer
	LR     float64 // learning rate
}

// Создание сети
func NewNetwork(sizes []int, lr float64) *Network {
	rand.Seed(time.Now().UnixNano())

	layers := make([]Layer, len(sizes)-1)

	for i := 0; i < len(sizes)-1; i++ {
		layers[i] = *NewLayer(sizes[i], sizes[i+1])
	}

	return &Network{
		Layers: layers,
		LR:     lr,
	}
}

// ===============================
// Forward pass (полный, с логами)
// ===============================

func (net *Network) ForwardFull(input []float64, training bool) ([]float64, [][]float64, [][]float64) {
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
			a2[j] = leakyReLU(sum)
		}

		// dropout (inverted)
		if training && layer.DropoutProb > 0 {
			p := layer.DropoutProb
			scale := 1.0 / (1.0 - p)

			for j := 0; j < layer.Out; j++ {
				if rand.Float64() < p {
					a2[j] = 0
				} else {
					a2[j] *= scale
				}
			}
		}

		layer.Z = z
		layer.A = a2

		activations = append(activations, a2)
		zvals = append(zvals, z)

		a = a2
	}

	return a, activations, zvals
}

// ===============================
// Backpropagation
// ===============================

func (net *Network) Backpropagate(target []float64, activations [][]float64, zvals [][]float64) [][]float64 {
	L := len(net.Layers)
	deltas := make([][]float64, L)

	// последний слой
	last := L - 1
	outAct := activations[last+1]
	outZ := zvals[last]

	deltas[last] = make([]float64, len(outAct))

	for i := range deltas[last] {
		deltas[last][i] = (outAct[i] - target[i]) * leakyReLUPrime(outZ[i])
	}

	// скрытые слои
	for l := L - 2; l >= 0; l-- {
		nextLayer := net.Layers[l+1]
		z := zvals[l]
		deltas[l] = make([]float64, len(z))

		for i := range z {
			sum := 0.0
			for k := range nextLayer.Biases {
				sum += nextLayer.Weights[k][i] * deltas[l+1][k]
			}
			deltas[l][i] = sum * leakyReLUPrime(z[i])
		}
	}

	return deltas
}

// ===============================
// Predict (без dropout)
// ===============================

func (net *Network) Predict(input []float64) []float64 {
	a := input

	for _, layer := range net.Layers {
		out := make([]float64, layer.Out)
		for i := 0; i < layer.Out; i++ {
			sum := layer.Biases[i]
			for j := 0; j < layer.In; j++ {
				sum += layer.Weights[i][j] * a[j]
			}
			out[i] = leakyReLU(sum)
		}
		a = out
	}

	return a
}

func (net *Network) ForwardDebug(input []float64) ([]float64, [][]float64, [][]float64) {
	return net.ForwardFull(input, false)
}

// ===============================
// Применение градиентов
// ===============================

func (net *Network) ApplyGradients(deltas [][]float64, activations [][]float64) {
	lr := net.LR

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

// ===============================
// Обучение с мини-батчами и ранней остановкой
// ===============================

func (net *Network) Train(samples, targets [][]float64, epochs int, batchSize int, f *os.File) {
	n := len(samples)

	bestLoss := 1e9
	wait := 0
	patience := 500  // количество эпох без улучшения
	minEpochs := 100  // минимальное число эпох перед ранней остановкой
	delta := 1e-8     // минимальное улучшение, чтобы считать loss значимым

	for e := 0; e < epochs; e++ {
		totalLoss := 0.0
		for i := 0; i < n; i++ {
			x := samples[i]
			y := targets[i]
			out, activations, zvals := net.ForwardFull(x, true)

			sampleLoss := 0.0
			for j := range y {
				diff := out[j] - y[j]
				sampleLoss += 0.5 * diff * diff
			}
			totalLoss += sampleLoss

			deltas := net.Backpropagate(y, activations, zvals)
			net.ApplyGradients(deltas, activations)
		}

		avgLoss := totalLoss / float64(n)

		if e%200 == 0 {
			fmt.Fprintf(f, "Epoch %d — loss=%.6f\n", e, avgLoss)
		}

		// === Ранняя остановка ===
		if e >= minEpochs {
			if bestLoss-avgLoss > delta {
				bestLoss = avgLoss
				wait = 0
			} else {
				wait++
				if wait >= patience {
					fmt.Fprintf(f, "Ранняя остановка на эпохе %d (нет улучшений %d эпох)\n", e, patience)
					break
				}
			}
		}
	}
}
